"""
Generate and store neighborhood text-profile embeddings using OpenAI's API.

Each neighborhood gets a short textual profile built from its features,
then embedded with text-embedding-3-small.  Embeddings are saved as .npy
for fast reload.

Usage (standalone):
    python -m src.embeddings          # embeds all neighborhoods, saves to outputs/embeddings/
    python -m src.embeddings --force   # re-embed even if cache exists
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "neighborhood_embeddings.npy"
TEXTS_PATH = EMBEDDINGS_DIR / "neighborhood_texts.npy"

# ── Text profile builder ────────────────────────────────────────────────────

def build_text_profile(row: pd.Series) -> str:
    """
    Compose a short natural-language profile for one neighborhood row.

    Columns used (from neighborhood_features_final.csv):
      neighborhood, borough, total_poi, category_entropy,
      avg_pedestrian, subway_station_count, poi_density_per_km2,
      commercial_activity_score, transit_activity_score
    """
    name = row.get("neighborhood", "Unknown")
    borough = row.get("borough", "")
    total_poi = int(row.get("total_poi", 0))
    entropy = round(float(row.get("category_entropy", 0)), 2)
    ped = int(row.get("avg_pedestrian", 0))
    subway = int(row.get("subway_station_count", 0))
    density = round(float(row.get("poi_density_per_km2", 0)), 1)
    commercial = round(float(row.get("commercial_activity_score", 0)), 0)
    transit = round(float(row.get("transit_activity_score", 0)), 0)
    nfh_overall = row.get("nfh_overall_score")
    nfh_shocks = row.get("nfh_goal4_fin_shocks_score")

    # Qualitative descriptors
    foot_traffic = (
        "very high" if ped > 5000
        else "high" if ped > 3000
        else "moderate" if ped > 1500
        else "low"
    )
    biz_density = (
        "extremely dense" if density > 30
        else "dense" if density > 10
        else "moderate" if density > 3
        else "sparse"
    )
    diversity = (
        "highly diverse" if entropy > 0.8
        else "diverse" if entropy > 0.6
        else "moderate" if entropy > 0.4
        else "limited"
    )
    nfh_txt = ""
    if pd.notna(nfh_overall):
        nfh_txt += f" Neighborhood financial health overall index score is {float(nfh_overall):.2f}."
    if pd.notna(nfh_shocks):
        nfh_txt += f" Financial-shock resilience score is {float(nfh_shocks):.2f}."

    return (
        f"{name} in {borough}. "
        f"{total_poi} points of interest with {biz_density} business density ({density}/km2). "
        f"Business category diversity is {diversity} (entropy {entropy}). "
        f"{foot_traffic} foot traffic (avg {ped} pedestrians). "
        f"{subway} subway stations nearby. "
        f"Commercial activity score {commercial}, transit activity score {transit}."
        f"{nfh_txt}"
    )


def build_all_profiles(df: pd.DataFrame) -> list[str]:
    """Return a text profile for every row in *df*."""
    return [build_text_profile(row) for _, row in df.iterrows()]


# ── OpenAI embedding ────────────────────────────────────────────────────────

def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    """
    Call OpenAI embeddings API and return (n, dim) float32 array.

    Batches automatically (API accepts up to 2048 inputs per call).
    """
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    vecs = [item.embedding for item in response.data]
    return np.array(vecs, dtype=np.float32)


# ── Persist / load ──────────────────────────────────────────────────────────

def save_embeddings(embeddings: np.ndarray, texts: list[str]) -> None:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(TEXTS_PATH, np.array(texts, dtype=object))


def load_embeddings() -> tuple[np.ndarray, list[str]] | None:
    """Return (embeddings, texts) if cache exists, else None."""
    if EMBEDDINGS_PATH.exists() and TEXTS_PATH.exists():
        emb = np.load(EMBEDDINGS_PATH)
        texts = np.load(TEXTS_PATH, allow_pickle=True).tolist()
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
        csv_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "processed"
            / "neighborhood_features_final.csv"
        )
    df = pd.read_csv(csv_path)
    texts = build_all_profiles(df)
    embeddings = embed_texts(texts)
    save_embeddings(embeddings, texts)
    return embeddings, texts


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    emb, texts = embed_neighborhood_features(force=force)
    print(f"Embedded {len(texts)} neighborhoods -> shape {emb.shape}")
    print(f"Saved to {EMBEDDINGS_PATH}")
    print(f"\nSample profile:\n{texts[0]}")
