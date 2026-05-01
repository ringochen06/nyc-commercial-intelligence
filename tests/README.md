# Tests

This directory contains two distinct categories of tests: **pytest unit tests** (fast, automated, no data required) and a **rank stability validation pipeline** (manual CLI scripts, require raw data files).

---

## Pytest unit tests

Run from the repo root:

```bash
pytest tests/test_feature_engineering.py tests/test_kmeans.py -q
# or use the shell helper:
bash tests/test.sh
```

`test.sh` auto-detects the local virtual environment (`.venv/bin/pytest`) and falls back to the system `pytest` if none is found.

### `test_kmeans.py`

Tests `src.kmeans_numpy` — the pure-NumPy K-means implementation (no sklearn clustering is used anywhere in that module).

| Test | What it checks |
|---|---|
| `test_kmeans_simple_clusters` | K-means++ on two well-separated 2D blobs produces correct label shape, centroid shape, and exactly 2 distinct labels |
| `test_kmeans_cache` | `kmeans_plus_plus_with_caching` writes results to HDF5 on the first call and returns `n_iter=0` (cache hit) on a second identical call |
| `test_assign_labels_nearest_centroid` | `assign_labels` maps each point to its geometrically closest centroid |
| `test_compute_inertia_matches_manual` | `compute_inertia` equals the hand-computed sum of squared distances (2.0 for the toy case) |
| `test_silhouette_well_separated_positive` | `silhouette_score` exceeds 0.3 for two well-separated blobs |
| `test_minibatch_kmeans_not_implemented` | `minibatch_kmeans` raises `NotImplementedError` (stub) |

### `test_feature_engineering.py`

Tests `storefront_activity_column_name` — the normalization function that maps raw DCA activity strings to `act_*_storefront` column names.

| Test | What it checks |
|---|---|
| `test_other_upper_maps_to_lower_other` | `"OTHER"` → `"act_other_storefront"` (case-preserving exception for the Supabase `OTHER`/`other` collision) |
| `test_whitespace_and_case_normalization` | Leading/trailing whitespace is stripped; standard activities are uppercased |
| `test_slashes_and_dashes_are_normalized` | `/` and `-` are replaced with `_` |
| `test_empty_activity_falls_back_to_unknown` | Empty string → `"act_UNKNOWN_storefront"` |

---

## Rank stability validation pipeline

These are **manual CLI scripts with no assertions**. They measure whether neighborhood rankings are stable across time by comparing a 2022 data snapshot against the present feature table.

### Step 1 — generate the 2022 snapshot

```bash
cd tests && python run_eval_pipeline.py
```

`run_eval_pipeline.py` orchestrates two sub-steps:

1. **`data_eval_processing.run_eval_processing(max_year=2022)`** — filters each raw data source to records on or before 2022:
   - Pedestrian counts: keeps only count columns whose year suffix ≤ 2022
   - Storefront filings: keeps rows whose `Reporting Year` max ≤ 2022
   - Neighborhood profiles and shooting incidents: filtered similarly
   - Outputs intermediate CSVs to `tests/data/2022/`: `ped_clean_test.csv`, `nbhd_clean_test.csv`, `storefront_features_test.csv`, `shooting_features_test.csv`

2. **Feature engineering on the 2022 data** — runs the same spatial join and aggregation logic as the production pipeline (`src.feature_engineering`) against the capped inputs. Output: `tests/data/2022/neighborhood_features_final_test.csv`.

### Step 2 — compare rankings across vintages

```bash
cd tests && python rank_stability_validation_business_queries.py
```

Default input paths:
- **Past (2022):** `tests/data/2022/neighborhood_features_final_test.csv`
- **Present:** `data/processed/neighborhood_features_final.csv`

The script embeds both feature tables using `src.embeddings` (respects `OPENAI_API_KEY` / `EMBEDDING_BACKEND`). Present-vintage embeddings use the standard `.npy` cache; past-vintage embeddings are cached separately as `*_past.npy` so the two never overwrite each other.

For each of 14 **default queries** (one per `act_*_storefront` business category — food services, retail, health care, finance, real estate, etc., phrased as natural-language site-selection prompts rather than column-name copies):

1. Compute a **blended score** for every CDTA in each vintage:

   ```
   score = α · MinMax(cosine_similarity) + (1 − α) · MinMax(−competitive_score)
   ```

   Default `α = 0.8`. `competitive_score` is `log1p(storefront_filing_count × avg_pedestrian)`.

2. Rank neighborhoods by blended score (rank 1 = highest).

3. Inner-join the two vintage rank tables on `(neighborhood, cd, borough)`.

4. Compute **Spearman rank correlation** (`r`) and **Kendall tau** (`τ`) between the 2022 and 2024 ranks.

### Outputs

All artifacts land in `outputs/validation/rank_stability_business_queries/` (relative to the repo root when invoked as above):

| File | Contents |
|---|---|
| `query_rank_correlations.csv` | One row per query: `query`, `n_cdta_overlap`, `spearman_r`, `kendall_tau` — sorted by `spearman_r` descending |
| `rank_compare_<slug>.csv` | Per-query per-CDTA table with 2022 rank, 2024 rank, `rank_delta`, `rank_delta_abs` — sorted by `rank_delta_abs` descending (largest movers first) |
| `rank_stability_rankings.html` | Single interactive Plotly scatter (2022 rank vs 2024 rank) with a dropdown to switch between queries; points colored by `|rank_delta|` |
| `ranking_stability_<slug>.html` | One standalone scatter per query |
| `query_rank_correlations_summary.html` | Bar + line chart of Spearman r and Kendall tau across all queries |

### CLI flags

```
--features-2022 PATH   Path to 2022 feature table (default: tests/data/2022/neighborhood_features_final_test.csv)
--features-2024 PATH   Path to present feature table (default: data/processed/neighborhood_features_final.csv)
--output-dir    PATH   Output directory (default: outputs/validation/rank_stability_business_queries)
--queries Q [Q ...]    Override DEFAULT_QUERIES with an explicit list
--derive-queries       Derive queries from common act_*_storefront column names instead
                       (note: inflates correlations because vocab mirrors embedded profiles)
--alpha FLOAT          Semantic weight (default: 0.8)
--clean-output         Delete existing files in output dir before writing
```

### Known gaps
- `--derive-queries` produces higher correlations than `DEFAULT_QUERIES` because the query vocabulary directly mirrors the `act_*` column names used to build the embedded text profiles.
