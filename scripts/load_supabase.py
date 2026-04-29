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


# Columns that exist in `public.neighborhoods` after migrations 0001-0007.
# 0006 dropped all per-activity (act_*) and NFH score columns since they're not
# consumed by /api/rank. The pipeline still computes them — they go into the
# embedding text profile — but we don't persist them in the table itself.
# Keep this list in sync with supabase/migrations/000{2,6}.
DB_COLUMNS_ALLOWED: set[str] = {
    "neighborhood",
    "cd",
    "borough",
    "area_km2",
    "avg_pedestrian",
    "peak_pedestrian",
    "pedestrian_count_points",
    "subway_station_count",
    "storefront_filing_count",
    "construction_jobs",
    "manufacturing_jobs",
    "wholesale_jobs",
    "pop_black",
    "pop_hispanic",
    "pop_asian",
    "total_population_proxy",
    "food_services",
    "total_businesses",
    "commute_public_transit",
    "pct_bachelors_plus",
    "total_jobs",
    "category_diversity",
    "category_entropy",
    "subway_density_per_km2",
    "storefront_density_per_km2",
    "commercial_activity_score",
    "transit_activity_score",
    # Added in 0006:
    "shooting_incident_count",
    "median_household_income",
    "competitive_score",
    # Re-added in 0008:
    "nfh_median_income",
    "nfh_poverty_rate",
    "nfh_pct_white",
    "nfh_pct_black",
    "nfh_pct_asian",
    "nfh_pct_hispanic",
    "nfh_goal1_fin_services_score",
    "nfh_goal2_goods_services_score",
    "nfh_goal3_jobs_income_score",
    "nfh_goal4_fin_shocks_score",
    "nfh_goal5_build_assets_score",
    "nfh_overall_score",
    # Populated per row by this script:
    "embedding",
    "embedding_text",
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
    # Look for credentials in either .env or .env.local (preferring .env if both exist).
    if (REPO_ROOT / ".env").is_file():
        load_dotenv(REPO_ROOT / ".env")
    if (REPO_ROOT / ".env.local").is_file():
        load_dotenv(REPO_ROOT / ".env.local")

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
    # Pandas widens int columns to float64 when any value is NaN, so the CSV
    # delivers e.g. 102.0 for subway_station_count. Postgres rejects floats
    # for integer columns ("invalid input syntax for type integer: 102.0"),
    # so coerce whole-number floats to ints. Ints serialize fine into both
    # integer and double-precision columns.
    db_df = to_db_columns(df)
    # Skip CSV columns the table doesn't have (act_*, nfh_*, etc.) — they're
    # still useful in the embedding text but the table schema doesn't store them.
    csv_cols = set(db_df.columns)
    persisted = csv_cols & DB_COLUMNS_ALLOWED
    skipped = sorted(csv_cols - DB_COLUMNS_ALLOWED)
    if skipped:
        print(f"skipping {len(skipped)} CSV columns not in the DB schema: {skipped[:6]}{'…' if len(skipped) > 6 else ''}")

    rows: list[dict] = []
    for i, row in db_df.iterrows():
        record: dict = {}
        for col, val in row.items():
            if col not in persisted:
                continue
            if pd.isna(val):
                record[col] = None
                continue
            if isinstance(val, (np.integer,)):
                record[col] = int(val)
            elif isinstance(val, (np.floating,)):
                f = float(val)
                record[col] = int(f) if f.is_integer() else f
            elif isinstance(val, float):
                record[col] = int(val) if val.is_integer() else val
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
