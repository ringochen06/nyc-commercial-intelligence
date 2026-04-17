# NYC Commercial Intelligence

A data-driven decision-support system for exploring and ranking commercial locations in New York City using urban data and machine learning.

---

## Overview

This project integrates NYC Open Data (business licenses, restaurant inspections, pedestrian counts, subway stations) and **NYC Public Neighborhood Profiles**–style community statistics, aggregated to **CDTA** boundaries, to model neighborhood-level commercial environments.

**`streamlit run app.py`** opens **K-Selection / clustering** (`app.py`). The **Ranking** UI is **`pages/Ranking.py`**: **hard SQL filters**, then **α·semantic + β·commercial_activity** (MinMax on the filtered rows; one **α** slider). Optional **Claude**.

### Data sources

Features are built from public NYC/NYS datasets, joined to **CDTA 2020** polygons:

- **POI / commerce:** DOHMH restaurant inspections; NYC issued business licenses; optional Kaggle legally operating businesses (see `run_pipeline.py`).
- **Mobility:** MTA subway stations; NYC DOT bi-annual pedestrian counts.
- **Community context:** NYC Comptroller Neighborhood Economic Profiles (ACS).
- **Optional:** Neighborhood Financial Health export → `nfh_*` columns when the file is included.

Filenames and URLs: **`data/raw/README.MD`**.

---

## End-to-end data flow

1. **Raw data** in `data/raw/` (CSVs + CDTA shapefile under `nyc_boundaries/`). See `data/raw/README.MD`.
2. **`python run_pipeline.py`** — `src/data_processing.py` cleans sources (including optional **Neighborhood Financial Health** / NFH CSV merged into `nbhd_clean` when the file is present); `src/feature_engineering.py` spatially aggregates to CDTA, merges **MOCEJ-style neighborhood profiles** and **`nfh_*` columns** on a normalized Community District key, then **imputes** remaining gaps in those merged numeric columns with **borough median, then citywide median** (dashboard-friendly proxy where a CDTA does not match a single profile row). **`commercial_activity_score`** = `total_poi` × `avg_pedestrian` and **`transit_activity_score`** = `subway_station_count` × `avg_pedestrian`, computed **after** filling missing POI/subway/pedestrian inputs so scores are not stuck at zero from ordering alone. Output: **`data/processed/neighborhood_features_final.csv`**. A healthy run ends with **no missing values** in that table; if any column still has NaN, investigate before shipping.
3. **Embeddings (for the app)** — `python -m src.embeddings` builds OpenAI embeddings from neighborhood text profiles; caches under `outputs/embeddings/`. Requires `OPENAI_API_KEY`. Use **`--force`** after changing features or profile text so embeddings match the CSV.
4. **`streamlit run app.py`** — **Home = K-Selection / clustering** (`app.py`); **Ranking** is **`pages/Ranking.py`** (hard filters, semantic blend, map). Loads the feature table (cached; **Rerun** or **Clear cache** after regenerating the CSV).

---

## Streamlit: Ranking page (`pages/Ranking.py`)

The ranking dashboard reads **`data/processed/neighborhood_features_final.csv`** (cached).

### 1. Hard filters (deterministic)

Sidebar controls set thresholds on:

- **Borough** (multiselect)
- **Minimum** `subway_station_count`, `avg_pedestrian`, `poi_density_per_km2`, `total_poi`, `commercial_activity_score`
- **Optional NFH** (when `nfh_overall_score` / `nfh_goal4_fin_shocks_score` exist): minimum thresholds via sidebar sliders

These are applied with **DuckDB**: the full table is registered as `nbhd`, a `SELECT … WHERE …` runs, and rows are ordered by **`commercial_activity_score` DESC**. The main area shows a table of surviving neighborhoods (key columns). **View generated SQL** expands to show the exact query. An expander (**About zeros, nulls, and refreshing data**) documents imputation, score formulas, and when zeros are expected.

If no rows match, the app stops with a warning.

### 2. Soft preferences — two-way ranking (in-app “ranking”)

- User enters a **free-text** query (ideal area description).
- **One blend slider** sets **α ∈ [0, 1]** for **semantic similarity** (cosine similarity after MinMax on the filtered set). **β = 1 − α** applies to the **MinMax-scaled** **`commercial_activity_score`** column. No second slider; **α + β = 1** by construction.
- **Embeddings:** query and neighborhoods use OpenAI **`text-embedding-3-small`** (`src/embeddings.py`). **Cosine similarity** is computed on the filtered set (aligned by neighborhood name to the full embedding matrix).
- Build a matrix **\[cosine_sim, commercial_activity_score\]** for those rows and apply **`sklearn.preprocessing.MinMaxScaler`** (column-wise, **0–1** on the filtered set). With a **single** row, scaling falls back to a neutral mid-score to avoid degenerate MinMax.
- **`blended_score = α·col0 + β·col1`**. Sort by **`blended_score`** descending. The table shows **`semantic_similarity`**, **`commercial_activity_score`**, and **`blended_score`**. Use the dataframe **download** control in the UI to export the ranking.
- **Map (when embeddings succeed):** a **CDTA choropleth** colors polygons by **`blended_score`** (sequential greens; requires `data/raw/nyc_boundaries/nycdta2020.shp`).

If embeddings are missing or the API key is unset, this block shows a warning (pre-generate embeddings with `python -m src.embeddings`; use **`--force`** after feature or profile text changes).

### 3. AI analysis (optional)

A button sends **Claude** a prompt with the soft query and the **hard-filtered** dataframe. The agent may call **`run_sql`** (read-only `SELECT` on the filtered data) and returns a natural-language recommendation (top neighborhoods + reasoning). Requires **`ANTHROPIC_API_KEY`**.

### 4. What is *not* in the Streamlit ranker

- **K-means clustering** — not used to order results in the app (it feeds optional cluster labels on the Ranking page only after you run K-Selection on the home page).

---

## Clustering vs ranking

| Piece | Role |
|-------|------|
| **K-means** + **K-Selection** (`app.py`, home) | **Exploratory:** sweep *k*, charts, CDTA choropleth, cluster briefs; labels saved for **Ranking**. **Does not** define the rank order on the Ranking page. |
| **Ranking** (`pages/Ranking.py`) | **Hard SQL filters** → **MinMax([semantic, commercial_activity])** → **α·col0 + (1−α)·col1** (optional map + Claude + optional cluster columns). |

---

## Optional supervised ML

**You do not have to use ML** for this dashboard: the baseline is transparent filters plus scores from the feature table. If you later add labeled outcomes, supervised models (with proper validation to avoid geographic leakage) can sit alongside the same pipeline outputs.

---

## Key components (code map)

| Area | Files | Notes |
|------|--------|--------|
| Data pipeline | `src/data_processing.py`, `src/feature_engineering.py`, `run_pipeline.py` | Produces `neighborhood_features_final.csv` |
| Embeddings | `src/embeddings.py` | OpenAI text profiles → `.npy` cache |
| K-Selection / clustering (home) | `app.py` | K sweep, viz, CDTA map; uses `src/kmeans_numpy.py`; writes cluster labels to session state |
| Ranking | `pages/Ranking.py` | Hard filters, MinMax blend (**one α**), optional map, Claude, optional cluster join |
| K-means (library) | `src/kmeans_numpy.py` | Used by `app.py`; **not** the ranking sort key |
| Agent | `src/agent.py` | Claude + DuckDB `SELECT` tools |
| Audit / one-off | `scripts/audit_imputation_fraction.py` | Replays merges to summarize imputation rates (optional) |

---

## Project Structure

```
data/        raw and processed datasets (see data/*/README*)
src/         core logic (pipeline, features, embeddings, agent)
pages/       `Ranking.py` only (soft ranker; **K-Selection lives in `app.py`**)
scripts/     optional maintenance scripts (e.g. imputation audit)
outputs/     saved models, embeddings, figures
tests/       unit tests (`pytest tests/`)
app.py       Streamlit entry: **K-Selection / clustering** home (`streamlit run app.py`)
```

### README index

| Document | Purpose |
|----------|---------|
| **`README.md`** (this file) | Setup, Streamlit behavior, API keys, troubleshooting |
| **`data/raw/README.MD`** | Where to obtain raw CSVs and CDTA shapefile; layout under `data/raw/` |
| **`data/processed/README.md`** | Processed CSVs, final feature columns, app + embedding pipeline |

---

## Setup

### Recommended (uv)

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Fallback (pip)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Build features and embeddings, then run the app

```bash
python run_pipeline.py
python -m src.embeddings    # requires OPENAI_API_KEY; use --force to refresh
streamlit run app.py        # optional: ANTHROPIC_API_KEY for Claude panel
```

Copy **`.env.example`** to **`.env`** and set API keys as needed (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, optional `OPENAI_EMBEDDING_MODEL`).

---

## Example queries (soft text)

- "quiet residential area for boutique retail"
- "high density food area"
- "stable neighborhood for cafes"

---

## Algorithm implementation (course / clustering)

K-means is implemented from scratch in `src/kmeans_numpy.py` (Euclidean distance, iterative centroids). The **K-Selection** home page (`app.py`) runs sweeps and charts; **Ranking** (`pages/Ranking.py`) is separate.

---

## Testing

```bash
pytest tests/
```

Includes **`tests/test_kmeans.py`** and **`tests/test_feature_engineering.py`**. Optional scripts under **`scripts/`** (e.g. imputation audit) are not run by CI.

---

## Data & live demo

- **`data/processed/`** is **committed** so you can run **`streamlit run app.py`** without rebuilding features. Re-run **`python run_pipeline.py`** after changing pipeline code or raw inputs.
- **`data/raw/`** CSVs are **not committed** (download locally; see `data/raw/README.MD`). The **CDTA 2020 shapefile** under **`data/raw/nyc_boundaries/nycdta2020.*`** **is committed** (~1.5MB) so spatial joins work out of the box. Large sources (e.g. restaurant inspections) stay local.
- **Regenerate processed tables:** `python run_pipeline.py` (requires **`geopandas`**, local CSVs as above, and the repo shapefile path).

## Notes

- Large datasets are not included in the repository.
- Precomputed embeddings live under `outputs/embeddings/` after running `python -m src.embeddings` (typically `neighborhood_embeddings.npy` and `neighborhood_texts.npy`).
- OpenAI **429 / insufficient_quota** means the account billing or quota for that API key is exhausted; fix billing in the OpenAI dashboard, then re-run embeddings.
