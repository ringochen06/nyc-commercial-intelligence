# NYC Commercial Intelligence

A data-driven decision-support system for exploring and ranking commercial locations in New York City using urban data and machine learning.

---

## Overview

This project integrates NYC Open Data (business licenses, restaurant inspections, pedestrian counts, subway stations) and **NYC Public Neighborhood Profiles**–style community statistics, aggregated to **CDTA** boundaries, to model neighborhood-level commercial environments.

The **Streamlit app** (`app.py`) ranks neighborhoods using **hard filters (DuckDB SQL)** plus a **two-way soft score**: **α·semantic + β·commercial_activity**. For each run, **cosine similarity** (query vs neighborhood text profiles) and **`commercial_activity_score`** are **MinMax-scaled to [0,1] on the filtered rows** so neither dominates by raw scale. The UI uses **one sidebar slider** for **α** (weight on semantic similarity); **β = 1 − α** is the weight on the scaled commercial-activity column. Optional **Claude** analysis runs read-only SQL on the filtered table. **Clustering** (K-means) is for **interpretation and exploration**, not for the in-app ranking. **`src/persistence_model.py` / `src/ranking.py`** are optional **supervised** scaffolds for future labeled tasks.

---

## End-to-end data flow

1. **Raw data** in `data/raw/` (CSVs + CDTA shapefile under `nyc_boundaries/`). See `data/raw/README.MD`.
2. **`python run_pipeline.py`** — `src/data_processing.py` cleans sources; `src/feature_engineering.py` spatially aggregates to CDTA and merges neighborhood profile columns on a normalized Community District key → **`data/processed/neighborhood_features_final.csv`**. The table includes **`commercial_activity_score`** (and **`transit_activity_score`**, etc.). **Profile columns** (demographics, jobs, etc.) may be **NaN** for some CDTAs where the profile CSV has no matching row; that is a **coverage / join** limitation, not necessarily corrupt raw files. The pipeline prints NaN counts with that expectation.
3. **Embeddings (for the app)** — `python -m src.embeddings` builds OpenAI embeddings from neighborhood text profiles; caches under `outputs/embeddings/`. Requires `OPENAI_API_KEY`.
4. **`streamlit run app.py`** — loads the feature table and runs the logic below.

---

## Streamlit app logic (`app.py`)

The dashboard reads **`data/processed/neighborhood_features_final.csv`** (cached).

### 1. Hard filters (deterministic)

Sidebar controls set thresholds on:

- **Borough** (multiselect)
- **Minimum** `subway_station_count`, `avg_pedestrian`, `poi_density_per_km2`, `total_poi`, `commercial_activity_score`

These are applied with **DuckDB**: the full table is registered as `nbhd`, a `SELECT … WHERE …` runs, and rows are ordered by **`commercial_activity_score` DESC**. The main area shows a table of surviving neighborhoods (key columns). **View generated SQL** expands to show the exact query.

If no rows match, the app stops with a warning.

### 2. Soft preferences — two-way ranking (in-app “ranking”)

- User enters a **free-text** query (ideal area description).
- **One blend slider** sets **α ∈ [0, 1]** for **semantic similarity** (cosine similarity after MinMax on the filtered set). **β = 1 − α** applies to the **MinMax-scaled** **`commercial_activity_score`** column. No second slider; **α + β = 1** by construction.
- **Embeddings:** query and neighborhoods use OpenAI **`text-embedding-3-small`** (`src/embeddings.py`). **Cosine similarity** is computed on the filtered set (aligned by neighborhood name to the full embedding matrix).
- Build a matrix **\[cosine_sim, commercial_activity_score\]** for those rows and apply **`sklearn.preprocessing.MinMaxScaler`** (column-wise, **0–1** on the filtered set). With a **single** row, scaling falls back to a neutral mid-score to avoid degenerate MinMax.
- **`blended_score = α·col0 + β·col1`**. Sort by **`blended_score`** descending. The table shows **`semantic_similarity`**, **`commercial_activity_score`**, and **`blended_score`**. Use the dataframe **download** control in the UI to export the ranking.

If embeddings are missing or the API key is unset, this block shows a warning (pre-generate embeddings with `python -m src.embeddings`; use **`--force`** after feature or profile text changes).

### 3. AI analysis (optional)

A button sends **Claude** a prompt with the soft query and the **hard-filtered** dataframe. The agent may call **`run_sql`** (read-only `SELECT` on the filtered data) and returns a natural-language recommendation (top neighborhoods + reasoning). Requires **`ANTHROPIC_API_KEY`**.

### 4. What is *not* in the Streamlit ranker

- **K-means clustering** — not used to order results in the app.
- **`src/ranking.py`** — not wired to `app.py` (the app implements blending inline).
- **`src/persistence_model.py`** — not wired to the app.

---

## Clustering vs ranking

| Piece | Role |
|-------|------|
| **K-means** (`src/kmeans_numpy.py`) | **Explanatory / exploratory** (e.g. group CDTA rows in feature space, course assignment). **Does not** feed the Streamlit ranking. |
| **Streamlit ranking** | **Hard SQL filters** → **MinMax([semantic, commercial_activity])** → **α·col0 + (1−α)·col1** (plus optional Claude narrative). |

---

## Optional supervised models and ML

- **You do not have to use ML.** A transparent baseline (rules, rates by borough/category, or a simple score from aggregates) can be enough for prototypes or reporting.
- **ML (e.g. Ridge / Random Forest in `src/persistence_model.py`)** becomes useful when you have **enough labeled examples** (neighborhoods or tracts with a clear outcome), want **out-of-sample generalization**, and can handle **validation** (e.g. time-based splits) to avoid leaking geography.

So: **ML is optional**; use it when labels and validation justify a predictive model, not as a prerequisite for the rest of the project.

---

## Key components (code map)

| Area | Files | Notes |
|------|--------|--------|
| Data pipeline | `src/data_processing.py`, `src/feature_engineering.py`, `run_pipeline.py` | Produces `neighborhood_features_final.csv` |
| Embeddings | `src/embeddings.py` | OpenAI text profiles → `.npy` cache |
| Dashboard | `app.py` | Hard filters, MinMax blend with **one α slider** (β = 1 − α), Claude |
| Clustering (standalone) | `src/kmeans_numpy.py` | Explanation / analysis; **not** app ranking |
| Supervised persistence (optional) | `src/persistence_model.py`, `src/ranking.py` | Scaffolds; **not** used by `app.py` |
| Agent | `src/agent.py` | Claude + DuckDB `SELECT` tools |

---

## Project Structure

```
data/        raw and processed datasets (see data/*/README*)
src/         core logic (pipeline, features, models, ranking)
outputs/     saved models, embeddings, figures
tests/       unit tests
app.py       Streamlit entry point
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

K-means is specified from scratch in `src/kmeans_numpy.py` (Euclidean distance, iterative centroids, convergence). Used for **clustering analysis**, **not** for the Streamlit ranking pipeline.

---

## Testing

```bash
pytest tests/
```

Includes skipped scaffolds for K-means and persistence labels until those modules are completed.

---

## Data & live demo

- **`data/raw/`** and **`data/processed/`** are **not committed** by default (see `.gitignore`). Cloning GitHub gives you code only, not CSVs or shapefiles.
- **Prepare data:** follow `data/raw/README.MD` (e.g. Hugging Face `ringoch/nyc-commercial-data` or NYC Open Data). Filenames must match `run_pipeline.py`.
- **Build processed tables:** `python run_pipeline.py` writes `data/processed/neighborhood_features_final.csv` (requires **`geopandas`** among deps).
- **Demo on a new machine:** copy `data/raw` (and optionally `data/processed`) from USB / Hugging Face / zip — **do not assume** data is on GitHub.

## Notes

- Large datasets are not included in the repository.
- Precomputed embeddings live under `outputs/embeddings/` after running `python -m src.embeddings` (typically `neighborhood_embeddings.npy` and `neighborhood_texts.npy`).
- OpenAI **429 / insufficient_quota** means the account billing or quota for that API key is exhausted; fix billing in the OpenAI dashboard, then re-run embeddings.
