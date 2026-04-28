"""One-shot loader: push neighborhood_features_final.csv + OpenAI embeddings into Supabase.

Reads the feature CSV, builds OpenAI text-embedding-3-small vectors, and
upserts into ``public.neighborhoods`` using the service-role key.

Required environment (in .env or shell):
    SUPABASE_URL                 e.g. https://<ref>.supabase.co
    SUPABASE_SERVICE_ROLE_KEY    server-only, bypasses RLS
    OPENAI_API_KEY               for the embedding call

Optional:
    EMBEDDINGS_FORCE=1           rebuild embeddings even if outputs/embeddings/ has a cache
    BATCH=100                    upsert batch size

Run:
    uv run python scripts/load_supabase.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.embeddings import (  # noqa: E402
    build_all_profiles,
    embed_texts,
    load_embeddings,
    save_embeddings,
)


CSV_PATH = REPO_ROOT / "data" / "processed" / "neighborhood_features_final.csv"


# CSV → DB column rename. Postgres folds identifiers to lowercase, so the two CSV
# columns that differ only in case (act_OTHER_* vs act_other_*) collide. The DB
# schema disambiguates by suffixing the lowercase variant with _lower.
RENAME_MAP: dict[str, str] = {
    "act_other_storefront": "act_other_lower_storefront",
    "act_other_density": "act_other_lower_density",
}


def to_db_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase column names; resolve the act_OTHER_/act_other_ collision."""
    out = df.copy()
    # First rename the originally-lowercase columns to *_lower_*.
    out = out.rename(columns=RENAME_MAP)
    # Then lowercase everything else (turns act_OTHER_storefront → act_other_storefront).
    out.columns = [c.lower() for c in out.columns]
    return out


def main() -> None:
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not service_key:
        raise SystemExit(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set (for the embedding call).")

    if not CSV_PATH.is_file():
        raise SystemExit(f"feature CSV missing: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"loaded {len(df)} rows from {CSV_PATH}")

    texts = build_all_profiles(df)

    # Either reuse a cached embedding matrix or compute fresh.
    cached = None if os.getenv("EMBEDDINGS_FORCE") else load_embeddings(backend="openai")
    if cached is not None and cached[0].shape[0] == len(df):
        emb, _ = cached
        print(f"using cached embeddings: shape {emb.shape}")
    else:
        print("calling OpenAI text-embedding-3-small for all rows...")
        emb = embed_texts(texts)
        save_embeddings(emb, texts, backend="openai")
        print(f"saved cache: shape {emb.shape}")

    if emb.shape[1] != 1536:
        raise SystemExit(
            f"embedding dim {emb.shape[1]} != 1536 (the schema's vector(1536)). "
            "Recreate the schema with the right dim or use OpenAI text-embedding-3-small."
        )

    # Build the row payload.
    db_df = to_db_columns(df)
    rows: list[dict] = []
    for i, row in db_df.iterrows():
        record = {}
        for col, val in row.items():
            if pd.isna(val):
                record[col] = None
            elif isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                record[col] = float(val)
            else:
                record[col] = val
        record["embedding"] = emb[i].tolist()
        record["embedding_text"] = str(texts[i])
        rows.append(record)

    # supabase-py is lazy-imported so the rest of the project doesn't need it.
    from supabase import create_client

    client = create_client(supabase_url, service_key)

    batch = int(os.getenv("BATCH", "50"))
    total = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        client.table("neighborhoods").upsert(chunk, on_conflict="neighborhood").execute()
        total += len(chunk)
        print(f"  upserted {total}/{len(rows)}")

    print(f"done. {len(rows)} neighborhoods loaded into public.neighborhoods.")


if __name__ == "__main__":
    main()
