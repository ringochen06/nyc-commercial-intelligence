"""
Generate and store neighborhood text-profile embeddings.

Backends (see EMBEDDING_BACKEND):
  - openai: text-embedding-3-small (or OPENAI_EMBEDDING_MODEL) via API
  - sentence_transformers: local model (default all-MiniLM-L6-v2), no API
  - auto: OpenAI if OPENAI_API_KEY is set, else sentence-transformers

Caches under outputs/embeddings/:
  - OpenAI vectors: neighborhood_embeddings.npy (+ neighborhood_texts.npy)
  - Sentence-transformers: neighborhood_embeddings_st.npy (same texts file)

Usage (standalone):
    python -m src.embeddings          # embeds all neighborhoods, saves cache
    python -m src.embeddings --force   # re-embed even if cache exists
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

try:
    from config import NEIGHBORHOOD_FEATURES_CSV
except ImportError:
    from src.config import NEIGHBORHOOD_FEATURES_CSV

load_dotenv()

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
).strip()

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "neighborhood_embeddings.npy"
EMBEDDINGS_ST_PATH = EMBEDDINGS_DIR / "neighborhood_embeddings_st.npy"
TEXTS_PATH = EMBEDDINGS_DIR / "neighborhood_texts.npy"

_st_model = None


def _backend_env_raw() -> str:
    return os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()


def resolve_embedding_backend() -> str:
    """
    Effective backend for cache paths and embed_texts / embed_neighborhood_features.

    Values: "openai" | "sentence_transformers".
    """
    raw = _backend_env_raw()
    if raw in ("sentence_transformers", "st", "sbert"):
        return "sentence_transformers"
    if raw == "auto":
        if os.getenv("OPENAI_API_KEY", "").strip():
            return "openai"
        return "sentence_transformers"
    return "openai"


def _cache_paths(backend: str) -> tuple[Path, Path]:
    if backend == "sentence_transformers":
        return EMBEDDINGS_ST_PATH, TEXTS_PATH
    return EMBEDDINGS_PATH, TEXTS_PATH


def _get_sentence_transformer(model_name: str | None = None):
    global _st_model
    name = (model_name or SENTENCE_TRANSFORMER_MODEL).strip()
    if _st_model is not None and getattr(_st_model, "_ci_name", None) == name:
        return _st_model
    from sentence_transformers import SentenceTransformer

    _st_model = SentenceTransformer(name)
    setattr(_st_model, "_ci_name", name)
    return _st_model


def _embed_texts_sentence_transformers(
    texts: list[str], *, model_name: str | None = None
) -> np.ndarray:
    model = _get_sentence_transformer(model_name)
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=len(texts) > 128,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


# ── Text profile builder ────────────────────────────────────────────────────


def build_text_profile(row: pd.Series) -> str:
    """
    Compose a short natural-language profile for one neighborhood row.

    Columns used (from neighborhood_features_final.csv), when present:
      neighborhood, borough, area_km2,
      storefront_filing_count, storefront_density_per_km2,
      all ``act_*_storefront`` with count > 0 (sorted by count, written into prose),
      category_diversity, category_entropy,
      avg_pedestrian, subway_station_count,
      pop_black, pop_hispanic, pop_asian, total_population_proxy (MOCEJ-style counts; each spelled out separately for semantic search),
      nfh_median_income, pct_bachelors_plus, commute_public_transit,
      commercial_activity_score, transit_activity_score,
      nfh_overall_score, nfh_goal4_fin_shocks_score
    """
    name = row.get("neighborhood", "Unknown")
    borough = row.get("borough", "")
    area_km2 = round(float(row.get("area_km2", 0) or 0), 1)
    sf_count = int(row.get("storefront_filing_count", 0) or 0)
    cat_div = int(row.get("category_diversity", 0) or 0)
    entropy = round(float(row.get("category_entropy", 0)), 2)
    ped = int(row.get("avg_pedestrian", 0))
    subway = int(row.get("subway_station_count", 0))
    density = round(float(row.get("storefront_density_per_km2", 0)), 1)
    mhi = row.get("nfh_median_income")
    pct_bach = row.get("pct_bachelors_plus")
    commute_pt = row.get("commute_public_transit")
    commercial = round(float(row.get("commercial_activity_score", 0)), 0)
    competitive = round(float(row.get("competitive_score", 0)), 2)
    transit = round(float(row.get("transit_activity_score", 0)), 0)
    shooting_incidents = int(float(row.get("shooting_incident_count_2024", 0) or 0))
    construction_jobs = int(float(row.get("construction_jobs", 0) or 0))
    manufacturing_jobs = int(float(row.get("manufacturing_jobs", 0) or 0))
    wholesale_jobs = int(float(row.get("wholesale_jobs", 0) or 0))
    total_jobs = int(float(row.get("total_jobs", 0) or 0))
    food_services = int(float(row.get("food_services", 0) or 0))
    total_businesses = int(float(row.get("total_businesses", 0) or 0))
    nfh_overall = row.get("nfh_overall_score")
    nfh_shocks = row.get("nfh_goal4_fin_shocks_score")
    pop_b = row.get("pop_black")
    pop_h = row.get("pop_hispanic")
    pop_a = row.get("pop_asian")
    pop_tot = row.get("total_population_proxy")

    foot_traffic = (
        "very high"
        if ped > 5000
        else "high" if ped > 3000 else "moderate" if ped > 1500 else "low"
    )
    biz_density = (
        "extremely dense"
        if density > 30
        else "dense" if density > 10 else "moderate" if density > 3 else "sparse"
    )
    diversity = (
        "highly diverse"
        if entropy > 0.8
        else "diverse" if entropy > 0.6 else "moderate" if entropy > 0.4 else "limited"
    )
    nfh_txt = ""
    if pd.notna(nfh_overall):
        nfh_txt += f" Neighborhood financial health overall index score is {float(nfh_overall):.2f}."
    if pd.notna(nfh_shocks):
        nfh_txt += f" Financial-shock resilience score is {float(nfh_shocks):.2f}."

    mix_txt = f"{cat_div} distinct primary-business-activity buckets from storefront filings. "

    soc_parts: list[str] = []
    if pd.notna(mhi):
        try:
            soc_parts.append(f"NFH median income about ${int(float(mhi)):,}")
        except (TypeError, ValueError):
            pass
    if pd.notna(pct_bach):
        try:
            soc_parts.append(
                f"about {float(pct_bach):.0f}% adults with bachelor's or higher (community profile)"
            )
        except (TypeError, ValueError):
            pass
    soc_txt = (
        (" Community socioeconomic proxies: " + "; ".join(soc_parts) + ".")
        if soc_parts
        else ""
    )

    # Separate sentences so queries like "Asian restaurant" align Asian population with FOOD SERVICES counts.
    pop_demog_txt = ""
    if pd.notna(pop_tot) and float(pop_tot or 0) > 0:
        try:
            b = int(float(pop_b or 0))
            h = int(float(pop_h or 0))
            a = int(float(pop_a or 0))
            t = int(float(pop_tot))
            pop_demog_txt = (
                " Community demographics (MOCEJ-style resident counts by group, not percentages):"
                f" Black population about {b:,}; Hispanic population about {h:,}; Asian population about {a:,};"
                f" total population proxy (sum of those three groups) about {t:,}."
            )
        except (TypeError, ValueError):
            pop_demog_txt = ""

    comm_txt = ""
    if pd.notna(commute_pt):
        try:
            comm_txt = f" About {float(commute_pt):.0f}% of workers commute by public transit (community profile)."
        except (TypeError, ValueError):
            pass

    sf_txt = ""
    if pd.notna(sf_count) and float(sf_count or 0) > 0:
        try:
            act_cols = [
                c
                for c in row.index
                if str(c).startswith("act_")
                and str(c).endswith("_storefront")
                and float(row.get(c, 0) or 0) > 0
            ]
            pairs: list[tuple[str, float]] = []
            for c in act_cols:
                slug = (
                    str(c).removeprefix("act_").removesuffix("_storefront").replace("_", " ")
                )
                pairs.append((slug, float(row[c])))
            pairs.sort(key=lambda x: -x[1])
            # List every non-zero activity so free-text queries (e.g. a specific NAICS bucket)
            # can match CDTAs where that category is present even if it is not in the top few.
            top_txt = ", ".join(f"{n} ({int(v)})" for n, v in pairs) if pairs else ""
            sf_txt = (
                f" Non-vacant storefront filings by primary business activity: {int(float(sf_count))} total."
                + (f" Counts by activity: {top_txt}." if top_txt else "")
            )
        except (TypeError, ValueError):
            sf_txt = ""

    return (
        f"{name} in {borough}. "
        f"CDTA area about {area_km2} km2. "
        f"{sf_count} non-vacant storefront filings; {biz_density} filing density ({density}/km2). "
        f"{mix_txt}"
        f"Activity mix diversity is {diversity} (entropy {entropy}). "
        f"{foot_traffic} foot traffic (avg {ped} pedestrians). "
        f"{subway} subway stations nearby. "
        f"Commercial activity score {commercial}, competitive score {competitive}, transit activity score {transit}."
        f" 2024 shooting incident count {shooting_incidents}."
        f" Employment composition (counts): construction {construction_jobs}, manufacturing {manufacturing_jobs}, wholesale {wholesale_jobs}, total jobs {total_jobs}."
        f" Business stock proxies: food services {food_services}, total businesses {total_businesses}."
        f"{soc_txt}{comm_txt}"
        f"{pop_demog_txt}"
        f"{sf_txt}"
        f"{nfh_txt}"
    )


def build_all_profiles(df: pd.DataFrame) -> list[str]:
    """Return a text profile for every row in *df*."""
    return [build_text_profile(row) for _, row in df.iterrows()]


# ── OpenAI embedding ────────────────────────────────────────────────────────


def _embed_texts_openai(texts: list[str], model: str) -> np.ndarray:
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype=np.float32)


def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """
    Embed *texts* with the active backend (see resolve_embedding_backend).

    For OpenAI, *model* defaults to OPENAI_EMBEDDING_MODEL.
    For sentence-transformers, *model* defaults to SENTENCE_TRANSFORMER_MODEL.
    """
    backend = resolve_embedding_backend()
    if backend == "sentence_transformers":
        return _embed_texts_sentence_transformers(texts, model_name=model)
    openai_model = model or EMBEDDING_MODEL
    return _embed_texts_openai(texts, openai_model)


# ── Persist / load ──────────────────────────────────────────────────────────


def save_embeddings(
    embeddings: np.ndarray, texts: list[str], *, backend: str | None = None
) -> None:
    b = backend or resolve_embedding_backend()
    emb_path, texts_path = _cache_paths(b)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    np.save(texts_path, np.array(texts, dtype=object))


def load_embeddings(
    *, backend: str | None = None
) -> tuple[np.ndarray, list[str]] | None:
    """Return (embeddings, texts) if cache exists for *backend*, else None."""
    b = backend or resolve_embedding_backend()
    emb_path, texts_path = _cache_paths(b)
    if emb_path.exists() and texts_path.exists():
        emb = np.load(emb_path)
        texts = np.load(texts_path, allow_pickle=True).tolist()
        return emb, texts
    return None


# ── Cosine similarity ───────────────────────────────────────────────────────


def cosine_similarity(query_vec: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a corpus matrix."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
    return corpus_norm @ query_norm


# ── Main (CLI) ──────────────────────────────────────────────────────────────


def embed_neighborhood_features(
    csv_path: str | Path | None = None, *, force: bool = False
) -> tuple[np.ndarray, list[str]]:
    """
    End-to-end: load CSV -> build text profiles -> embed -> cache -> return.
    """
    if not force:
        cached = load_embeddings()
        if cached is not None:
            return cached

    if csv_path is None:
        csv_path = NEIGHBORHOOD_FEATURES_CSV
    df = pd.read_csv(csv_path)
    texts = build_all_profiles(df)
    backend = resolve_embedding_backend()

    if backend == "openai":
        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            if _backend_env_raw() == "auto":
                logger.warning(
                    "OpenAI embedding failed (%s); falling back to sentence-transformers.",
                    e,
                )
                embeddings = _embed_texts_sentence_transformers(texts)
                save_embeddings(embeddings, texts, backend="sentence_transformers")
                return embeddings, texts
            raise
        save_embeddings(embeddings, texts, backend="openai")
    else:
        embeddings = _embed_texts_sentence_transformers(texts)
        save_embeddings(embeddings, texts, backend="sentence_transformers")

    return embeddings, texts


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    emb, texts = embed_neighborhood_features(force=force)
    b = resolve_embedding_backend()
    path, _ = _cache_paths(b)
    print(f"Backend: {b}")
    print(f"Embedded {len(texts)} neighborhoods -> shape {emb.shape}")
    print(f"Saved to {path}")
    print(f"\nSample profile:\n{texts[0]}")
