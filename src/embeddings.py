"""
Generate and store neighborhood text-profile embeddings.

Each neighborhood gets a short textual profile built from its features.
The module prefers OpenAI `text-embedding-3-small`, and falls back to a
local `all-MiniLM-L6-v2` sentence-transformers model when OpenAI is not
available. Embeddings are saved as .npy for fast reload.

Usage (standalone):
    python -m src.embeddings           # embed all neighborhoods, save to outputs/embeddings/
    python -m src.embeddings --force   # re-embed even if cache exists
"""

from __future__ import annotations

import os
import json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT.parent
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
# Add multi-backend support: "auto" (default), "openai", or "local", where "auto" prefers OpenAI when available but falls back to local model if not.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "auto").strip().lower()
LOCAL_EMBEDDING_MODEL_DIR = Path(
    os.getenv("LOCAL_EMBEDDING_MODEL_DIR", str(WORKSPACE_ROOT / "all-MiniLM-L6-v2"))
)
EMBEDDINGS_DIR = REPO_ROOT / "outputs" / "embeddings"

_ACTIVE_EMBEDDING_BACKEND: str | None = None


def _set_active_backend(backend: str) -> None:
    """Remember which embedding backend is currently active in this process."""
    global _ACTIVE_EMBEDDING_BACKEND
    _ACTIVE_EMBEDDING_BACKEND = backend


def _cache_paths(backend: str) -> tuple[Path, Path, Path]:
    """Return backend-specific cache paths for embeddings, texts, and metadata."""
    backend_key = backend.strip().lower()
    if backend_key not in {"openai", "local"}:
        raise ValueError(f"Unsupported embedding backend: {backend}")
    return (
        EMBEDDINGS_DIR / f"neighborhood_embeddings_{backend_key}.npy",
        EMBEDDINGS_DIR / f"neighborhood_texts_{backend_key}.npy",
        EMBEDDINGS_DIR / f"embedding_metadata_{backend_key}.json",
    )


def _local_model_exists() -> bool:
    """Return whether the configured local MiniLM model directory exists."""
    return LOCAL_EMBEDDING_MODEL_DIR.exists()


def _openai_available() -> bool:
    """Return whether an OpenAI API key is present in the current environment."""
    return bool(os.getenv("OPENAI_API_KEY"))


def _import_sentence_transformer():
    """Import SentenceTransformer lazily so the local backend stays optional."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed.") from exc
    return SentenceTransformer


def _import_transformers_backend():
    """Import the lower-level Transformers stack used by the local fallback path."""
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Local embedding fallback requires either `sentence-transformers` "
            "or `transformers` + `torch`."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def _resolve_backend_from_metadata(
    embeddings: np.ndarray | None = None,
    metadata_path: Path | None = None,
) -> str:
    """Infer which backend produced a cached embedding matrix.

    This supports older caches that may not carry explicit backend metadata.
    In that case, vector width is used as a fallback hint.
    """
    if metadata_path is None:
        raise ValueError("metadata_path is required for backend-specific caches.")
    metadata_file = metadata_path
    if metadata_file.exists():
        data = json.loads(metadata_file.read_text(encoding="utf-8"))
        backend = str(data.get("backend", "")).strip().lower()
        if backend in {"openai", "local"}:
            return backend
    if embeddings is not None and embeddings.ndim == 2 and embeddings.shape[1] == 384:
        return "local"
    return "openai"


def _choose_backend() -> str:
    """Choose the backend to use for the current embedding request.

    auto mode prefers OpenAI when available and falls back to the local model when it is not.
    """
    requested = EMBEDDING_BACKEND
    if requested == "openai":
        return "openai"
    if requested == "local":
        return "local"

    if _ACTIVE_EMBEDDING_BACKEND == "openai" and not _openai_available():
        return "local" if _local_model_exists() else "openai"
    if _ACTIVE_EMBEDDING_BACKEND in {"openai", "local"}:
        return _ACTIVE_EMBEDDING_BACKEND

    if _openai_available():
        return "openai"
    if _local_model_exists():
        return "local"
    return "openai"


def _get_backend_request() -> str | None:
    """Return the explicit backend request, or None when auto-selection is enabled."""
    requested = EMBEDDING_BACKEND.strip().lower()
    if requested in {"openai", "local"}:
        return requested
    return None


def get_runtime_backend() -> str:
    """Return the embedding backend that should be used in this process."""
    return _choose_backend()

# ── Text profile builder ────────────────────────────────────────────────────


def build_text_profile(row: pd.Series) -> str:
    """
    Compose a short natural-language profile for one neighborhood row.

    Columns used (from neighborhood_features_final.csv):
      neighborhood, borough, area_km2, total_poi, category_diversity, ratio_retail,
      category_entropy, avg_pedestrian, subway_station_count, poi_density_per_km2,
      retail_density_per_km2, food_density_per_km2,
      nfh_median_income, pct_bachelors_plus, commute_public_transit,
      commercial_activity_score, transit_activity_score, optional nfh_* scores (no NFH ranks)
    """
    name = row.get("neighborhood", "Unknown")
    borough = row.get("borough", "")
    area_km2 = round(float(row.get("area_km2", 0) or 0), 1)
    total_poi = int(row.get("total_poi", 0))
    cat_div = int(row.get("category_diversity", 0) or 0)
    ratio_retail_v = float(row.get("ratio_retail", 0) or 0)
    entropy = round(float(row.get("category_entropy", 0)), 2)
    ped = int(row.get("avg_pedestrian", 0))
    subway = int(row.get("subway_station_count", 0))
    density = round(float(row.get("poi_density_per_km2", 0)), 1)
    retail_d = round(float(row.get("retail_density_per_km2", 0)), 2)
    food_d = round(float(row.get("food_density_per_km2", 0)), 2)
    mhi = row.get("nfh_median_income")
    pct_bach = row.get("pct_bachelors_plus")
    commute_pt = row.get("commute_public_transit")
    commercial = round(float(row.get("commercial_activity_score", 0)), 0)
    transit = round(float(row.get("transit_activity_score", 0)), 0)
    nfh_overall = row.get("nfh_overall_score")
    nfh_shocks = row.get("nfh_goal4_fin_shocks_score")

    # Qualitative descriptors
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

    # Retail share: ratio_retail is typically 0–1 (share of POIs)
    rr_pct = ratio_retail_v * 100.0 if ratio_retail_v <= 1.0 else ratio_retail_v

    mix_txt = f"{cat_div} simplified category groups; retail license share of POIs about {rr_pct:.0f}%. "

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

    comm_txt = ""
    if pd.notna(commute_pt):
        try:
            comm_txt = f" About {float(commute_pt):.0f}% of workers commute by public transit (community profile)."
        except (TypeError, ValueError):
            pass

    return (
        f"{name} in {borough}. "
        f"CDTA area about {area_km2} km2. "
        f"{total_poi} points of interest with {biz_density} business density ({density}/km2). "
        f"Retail-category POI density {retail_d}/km2; food-category POI density {food_d}/km2. "
        f"{mix_txt}"
        f"Business category diversity is {diversity} (entropy {entropy}). "
        f"{foot_traffic} foot traffic (avg {ped} pedestrians). "
        f"{subway} subway stations nearby. "
        f"Commercial activity score {commercial}, transit activity score {transit}."
        f"{soc_txt}{comm_txt}"
        f"{nfh_txt}"
    )


def build_all_profiles(df: pd.DataFrame) -> list[str]:
    """Return a text profile for every row in *df*."""
    return [build_text_profile(row) for _, row in df.iterrows()]


def _embed_texts_openai(texts: list[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a batch of texts with OpenAI and return a float32 matrix."""
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype=np.float32)


def _embed_texts_local(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts with the local MiniLM model.

    Prefer sentence-transformers when installed, otherwise fall back to raw
    Transformers + torch mean pooling over token embeddings.
    """
    if not _local_model_exists():
        raise RuntimeError(
            f"Local embedding model not found at {LOCAL_EMBEDDING_MODEL_DIR}."
        )
    try:
        SentenceTransformer = _import_sentence_transformer()
        model = SentenceTransformer(str(LOCAL_EMBEDDING_MODEL_DIR))
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception:
        # Some sentence-transformers / torch / transformers combinations can
        # fail at runtime with device-placement errors (for example `meta`
        # tensors). Fall back to the lower-level Transformers path instead of
        # treating local embeddings as unavailable.
        torch, AutoModel, AutoTokenizer = _import_transformers_backend()
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_EMBEDDING_MODEL_DIR))
        model = AutoModel.from_pretrained(str(LOCAL_EMBEDDING_MODEL_DIR))
        model.eval()

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            vecs = summed / counts
        return vecs.cpu().numpy().astype(np.float32, copy=False)


# ── Embedding backend selection ─────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    *,
    backend: str | None = None,
) -> np.ndarray:
    """
    Return a float32 embedding matrix for *texts*.

    The backend defaults to the active cached backend when available so the
    query vector and cached corpus vectors stay dimensionally aligned.
    """
    requested_backend = (backend or "").strip().lower()
    chosen_backend = requested_backend or _choose_backend()
    if chosen_backend == "local":
        vecs = _embed_texts_local(texts)
        _set_active_backend("local")
        return vecs

    try:
        vecs = _embed_texts_openai(texts, model=model)
        _set_active_backend("openai")
        return vecs
    except Exception:
        if requested_backend == "openai":
            raise
        vecs = _embed_texts_local(texts)
        _set_active_backend("local")
        return vecs


# ── Persist / load ──────────────────────────────────────────────────────────

def save_embeddings(embeddings: np.ndarray, texts: list[str], *, backend: str) -> None:
    """Persist embeddings, source texts, and backend metadata to disk."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path, texts_path, metadata_path = _cache_paths(backend)
    np.save(embeddings_path, embeddings)
    np.save(texts_path, np.array(texts, dtype=object))
    metadata_path.write_text(
        json.dumps(
            {
                "backend": backend,
                "openai_model": EMBEDDING_MODEL if backend == "openai" else None,
                "local_model_dir": str(LOCAL_EMBEDDING_MODEL_DIR) if backend == "local" else None,
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _set_active_backend(backend)


def load_embeddings(*, backend: str | None = None) -> tuple[np.ndarray, list[str]] | None:
    """Load cached embeddings/texts and restore the backend they belong to.

    When *backend* is provided, only return the cache if it matches the
    requested backend. This keeps callers that embed queries on the selected
    runtime backend aligned with the cached corpus vectors.
    """
    requested_backend = backend.strip().lower() if backend is not None else None

    candidate_paths: list[tuple[Path, Path, Path]] = []
    if requested_backend in {"openai", "local"}:
        candidate_paths.append(_cache_paths(requested_backend))
    else:
        if _ACTIVE_EMBEDDING_BACKEND in {"openai", "local"}:
            candidate_paths.append(_cache_paths(_ACTIVE_EMBEDDING_BACKEND))
        preferred_backend = _choose_backend()
        if preferred_backend in {"openai", "local"} and preferred_backend != _ACTIVE_EMBEDDING_BACKEND:
            candidate_paths.append(_cache_paths(preferred_backend))
        candidate_paths.extend(
            paths
            for paths in (_cache_paths("openai"), _cache_paths("local"))
            if paths not in candidate_paths
        )

    for embeddings_path, texts_path, metadata_path in candidate_paths:
        if not embeddings_path.exists() or not texts_path.exists():
            continue
        emb = np.load(embeddings_path)
        texts = np.load(texts_path, allow_pickle=True).tolist()
        cached_backend = _resolve_backend_from_metadata(emb, metadata_path=metadata_path)
        if requested_backend is not None and cached_backend != requested_backend:
            continue
        _set_active_backend(cached_backend)
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
    requested_backend = _get_backend_request()
    preferred_backend = requested_backend or _choose_backend()

    if not force:
        cached = load_embeddings(backend=preferred_backend)
        if cached is not None:
            return cached

    if csv_path is None:
        csv_path = NEIGHBORHOOD_FEATURES_CSV
    df = pd.read_csv(csv_path)
    texts = build_all_profiles(df)
    embeddings = embed_texts(texts, backend=requested_backend)
    active_backend = _ACTIVE_EMBEDDING_BACKEND or preferred_backend
    save_embeddings(embeddings, texts, backend=active_backend)
    return embeddings, texts


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    emb, texts = embed_neighborhood_features(force=force)
    backend = _ACTIVE_EMBEDDING_BACKEND or _choose_backend()
    embeddings_path, _, _ = _cache_paths(backend)
    print(f"Embedded {len(texts)} neighborhoods -> shape {emb.shape}")
    print(f"Saved to {embeddings_path}")
    print(f"Backend: {backend}")
    print(f"\nSample profile:\n{texts[0]}")
