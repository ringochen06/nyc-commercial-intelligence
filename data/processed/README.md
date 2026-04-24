# Processed Data

This folder contains cleaned and engineered datasets prepared for downstream analysis, embedding, clustering, and the Streamlit app.

---

## Files

### `ped_clean.csv`

Cleaned point-level pedestrian count dataset.

**Main fields**

- `borough`
- `street`
- `latitude`
- `longitude`
- `avg_pedestrian`
- `peak_pedestrian`

**Purpose**

- Represents pedestrian demand / foot traffic  
- Supports demand estimation and spatial analysis  

---

### `subway_clean.csv`

Cleaned point-level subway station dataset.

**Main fields**

- `station_name`
- `borough`
- `latitude`
- `longitude`
- `routes`

**Purpose**

- Represents transit accessibility  
- Supports accessibility and distance-based features  

---

### `storefront_features.csv`

Per-CDTA table written by **`run_feature_engineering`** — same storefront columns as merged into **`neighborhood_features_final.csv`** (`storefront_filing_count`, `act_*_storefront`, keyed by `neighborhood`, `cd`, `borough`). Built from the **raw** storefront Open Data CSV (in-memory clean + spatial join); not from any intermediate `storefront_clean` file.

---

### `nbhd_clean.csv`

Cleaned area-level socioeconomic dataset derived from **NYC Public Neighborhood Profiles** (MOCEJ-style column selection in `src/data_processing.py`), optionally merged with **Neighborhood Financial Health (NFH)** indicators when `data/raw/Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool_*.csv` is present and readable.

**Main fields (non-exhaustive)**

- `neighborhood`, `cd` (community district string, including combined districts such as `BX01 & BX02`), `borough`
- Job and business counts: `construction_jobs`, `manufacturing_jobs`, `wholesale_jobs`, `food_services`, `total_businesses`
- Demographics / SES proxies: `median_household_income`, `commute_public_transit` (**percent of 2016 employed** who commute via public transit — derived in `clean_neighborhood_profiles` from MOCEJ counts ÷ `2016 Employed`), `pct_bachelors_plus`, race population columns and derived `pct_*`, `total_jobs`, `total_population_proxy`
- **NFH (when merged):** `nfh_overall_score`, `nfh_goal4_fin_shocks_score`, other `nfh_goal*_score` / `*_rank`, and demographic columns such as `nfh_median_income`, `nfh_poverty_rate`, `nfh_pct_*` (see CSV header for the full list)

**Purpose**

- Joined onto the CDTA master table by a normalized district key (`normalize_cdta_join_key` in `src/data_processing.py`). NFH source rows use labels like `BX Community District 8`; these are normalized to codes such as `BX08` in the same module so they align with the same join logic as MOCEJ rows.

---

## Final Feature Table

### `neighborhood_features_final.csv`

- Written once by **`run_feature_engineering()`** in `src/feature_engineering.py` (also invoked from **`run_pipeline.py`**). This is the only merged feature table **`app.py`**, **`pages/Ranking.py`**, and **`src/embeddings.py`** read.

Each row is one **CDTA** (Community District Tabulation Area) polygon from `nycdta2020`. With MOCEJ profiles plus NFH merged, a typical build is **on the order of ~71 rows** (exact column count depends on optional NFH columns and pipeline version).

**Not in this table:** There is **no** `persistence_score` column in the current pipeline; the app blends **semantic similarity** and **`commercial_activity_score`** only (see below).

**`neighborhood_features_final.csv`** is built from **pedestrian** and **subway** point layers plus **storefront** aggregates (raw storefront CSV is read in `run_feature_engineering`; no `storefront_clean.csv` / `storefront_with_neighborhood.csv`). It includes per-CDTA **`storefront_filing_count`** (non-vacant filings only), **`storefront_density_per_km2`**, **`act_<CATEGORY>_storefront`** counts by primary business activity, and **`category_diversity` / `category_entropy`** derived from those counts. MOCEJ-derived **`pct_hispanic`**, **`pct_black`**, **`pct_asian`** are omitted from this merge (counts remain in **`pop_*`** and **`total_population_proxy`**). **`storefront_features.csv`** holds the storefront-only CDTA table (same storefront columns as merged into the final CSV).

Sources integrated:

- **Storefront** filings (optional raw path in `run_pipeline.py`), pedestrian counts, subway stations (aggregated after spatial join)  
- CDTA area (`area_km2`) and density / interaction features  
- **MOCEJ + NFH columns** from `nbhd_clean`, merged on the normalized district key. Where a CDTA still has no direct match after normalization, **numeric** profile and `nfh_*` columns are **imputed** in `merge_all_features` (borough median, then citywide median) so the final CSV has **no NaNs** and the Streamlit hard filters stay usable. Treat imputed values as **proxies**, not tract-level ground truth.

---

### Feature Groups

**Storefront (primary business activity)**

- `storefront_filing_count`, `storefront_density_per_km2`, `act_<CATEGORY>_storefront`
- `category_diversity` (number of non-zero activity buckets), `category_entropy` (mix across `act_*_storefront`)

**Pedestrian Activity**

- `avg_pedestrian`, `peak_pedestrian`
- `pedestrian_count_points`

**Transit Accessibility**

- `subway_station_count`
- `subway_density_per_km2`

**Interaction Features**

- `commercial_activity_score` = **`log1p`**(`storefront_filing_count` × `avg_pedestrian`) (after missing storefront counts are set to 0 and pedestrian averages use **borough median**, then **citywide median**, then **0**; the linear product is clipped at 0 before `log1p`). Used in **`app.py`** after MinMax scaling alongside semantic similarity.
- `transit_activity_score` = **`log1p`**(`subway_station_count` × `avg_pedestrian`) with the same pedestrian handling. Both scores are **0** when the inner product is 0 (e.g. no filings or no subway stations, or zero pedestrian signal after imputation).

**Geometry / area**

- `area_km2`

**Neighborhood profile (MOCEJ) and NFH**

Examples: `median_household_income`, `pct_bachelors_plus`, `commute_public_transit`, `total_businesses`, job counts, race population proxies; `nfh_overall_score`, `nfh_goal4_fin_shocks_score`, and other `nfh_*` columns when the NFH CSV was merged in `data_processing`. See the CSV header for the full list.

---

## Downstream: embeddings and Streamlit

1. **`python -m src.embeddings`** reads **`neighborhood_features_final.csv`**, builds one text profile per row (`src/embeddings.py`), then embeds with **OpenAI `text-embedding-3-small`** when `OPENAI_API_KEY` is set (and local-only is not forced), otherwise **sentence-transformers**; saves **`neighborhood_embeddings.npy`** (OpenAI) or **`neighborhood_embeddings_st.npy`** (local) plus shared **`neighborhood_texts.npy`**.  
2. **`streamlit run app.py`** — home is **K-Selection** (`app.py`); open **Ranking** (`pages/Ranking.py`) for filters and blend. Both load the CSV (cached) and embeddings (cached).  
3. **Ranking page — hard filters:** DuckDB `SELECT` with sidebar thresholds (borough, subway, pedestrian, storefront density, storefront filing count, **`commercial_activity_score`**, etc.).  
4. **Soft ranking:** On the filtered rows, **MinMaxScaler** is fit on **`[cosine_sim, commercial_activity_score]`**. One blend slider **α**; **β = 1 − α**. Output: **`blended_score`**. Optional **CDTA map** (choropleth by `blended_score`) when embeddings and the shapefile are available.  
5. Optional **NFH thresholds** when those columns exist.  
6. Optional **Claude** panel: read-only SQL on the filtered dataframe (`ANTHROPIC_API_KEY`).

Details and setup: **root `README.md`**.

---

### Text profile → embeddings (`src/embeddings.py`)

Each CDTA row is turned into one English paragraph (`build_text_profile`), then embedded (OpenAI **`text-embedding-3-small`** when a key is set, else **sentence-transformers**; override with **`EMBEDDING_BACKEND`**). After changing profile text or feature columns, rerun **`python -m src.embeddings --force`**.

**Soft / embedded columns (inputs to the text profile)**

| Column (or pattern) | Used in text profile |
|----------------------|----------------------|
| `neighborhood` | Area name |
| `borough` | Borough |
| `area_km2` | CDTA footprint (km²) |
| `storefront_filing_count` | Total non-vacant filings |
| `storefront_density_per_km2` | Filings per km² + qualitative density phrase |
| `act_<SLUG>_storefront` | One **numeric** column per Primary Business Activity from the storefront export. **Every** column with count **> 0** is written into the embedding profile (humanized activity label, sorted by count). Exact slugs match your CSV header (new Open Data categories can add columns). |
| `category_diversity` | Number of non-zero activity buckets |
| `category_entropy` | Mix / diversity across `act_*_storefront` |
| `avg_pedestrian` | Foot-traffic level + numeric average |
| `subway_station_count` | Transit access |
| `commercial_activity_score` | `log1p` storefront × pedestrian product |
| `transit_activity_score` | `log1p` subway × pedestrian product |
| `pop_black` | MOCEJ-style **Black** resident count (separate sentence in the profile) |
| `pop_hispanic` | MOCEJ-style **Hispanic** resident count (separate sentence) |
| `pop_asian` | MOCEJ-style **Asian** resident count (separate sentence; use with `act_FOOD_SERVICES_storefront` etc. for cuisine / “Asian restaurant”–style queries) |
| `total_population_proxy` | Sum of `pop_black` + `pop_hispanic` + `pop_asian` (separate sentence; not a census official total) |
| `nfh_median_income` | NFH median income (when present) |
| `pct_bachelors_plus` | Share with bachelor’s or higher |
| `commute_public_transit` | Public-transit commute share |
| `nfh_overall_score`, `nfh_goal4_fin_shocks_score` | NFH composite lines (when present) |
| **Blended** (Ranking UI only) | **Not** embedded: MinMax on **`[cosine_sim, commercial_activity_score]`** on the filtered rows, then **α·semantic + (1−α)·activity** |

**Example `act_*_storefront` column names** (one row per activity bucket in a typical Open Data export; yours may differ):

`act_ACCOUNTING_SERVICES_storefront`, `act_BROADCASTING_TELECOMM_storefront`, `act_EDUCATIONAL_SERVICES_storefront`, `act_FINANCE_AND_INSURANCE_storefront`, `act_FOOD_SERVICES_storefront`, `act_HEALTH_CARE_OR_SOCIAL_ASSISTANCE_storefront`, `act_INFORMATION_SERVICES_storefront`, `act_LEGAL_SERVICES_storefront`, `act_MANUFACTURING_storefront`, `act_MOVIES_VIDEO_SOUND_storefront`, `act_NO_BUSINESS_ACTIVITY_IDENTIFIED_storefront`, `act_OTHER_storefront`, `act_PUBLISHING_storefront`, `act_REAL_ESTATE_storefront`, `act_RETAIL_storefront`, `act_UNKNOWN_storefront`, `act_WHOLESALE_storefront`, `act_other_storefront`

The **Ranking** page table and “Soft / embedded columns” reference list these dynamically from `neighborhood_features_final.csv`.

**NFH `nfh_pct_*` columns:** If your build merged Neighborhood Financial Health, those percentage columns may still appear in the CSV for DuckDB / Claude. They are **not** fed into the embedding text profile (avoid duplicating or conflicting with MOCEJ **`pop_*`** counts used for semantic search).

---

### Data Integration Process

1. Raw datasets are cleaned (`src/data_processing.py`) → `*_clean.csv` under `data/processed/`.  
2. Point layers (pedestrian, subway; optional storefront from raw CSV) are spatially joined to CDTA boundaries; features are aggregated per CDTA in memory (`src/feature_engineering.py`). Intermediate `*_with_neighborhood.csv` / per-layer `*_features.csv` files are not written.  
3. **`nbhd_clean` is merged** onto the CDTA master table using a normalized district code (see `normalize_cdta_join_key` in `src/data_processing.py`).  
4. **Missing values (spatial / activity columns):** Storefront / subway counts and densities are filled with **0** where appropriate. **Pedestrian:** `avg_pedestrian` / `peak_pedestrian` use **borough median**, then **citywide median**, then **0** (avoids imputing every missing CDTA with one global mean); `pedestrian_count_points` missing → **0**.  
5. **Interaction scores** (`commercial_activity_score`, `transit_activity_score`) are **`log1p`** of the linear products, computed **after** those fills so they reflect imputed inputs and have gentler tails for filters / MinMax.  
6. **Profile + `nfh_*` numeric columns** after the merge: **borough median**, then **citywide median**, for any remaining NaN.  
7. **`run_pipeline.py`** prints a NaN summary; a **clean** build should report **no missing values** in the final feature table.

---

### Notes

- **Geography** is CDTA; neighborhood-profile and NFH data enter as **attributes**, not as the boundary source.  
- **Combined community districts** (e.g. `BX01 & BX02` in MOCEJ, or `BX Community Districts 1 & 2` in NFH) are normalized to a **single** code (first district number) for joining; that is an intentional mapping, not duplicate rows in the source.  
- Pedestrian coverage is limited by where sensors exist.  
- This folder’s CSVs are **committed in the repo** for convenience; regenerate anytime with **`python run_pipeline.py`** after changing code or raw inputs.  
- If OpenAI returns **429 / insufficient_quota** when embedding, fix billing or quota for `OPENAI_API_KEY`, then rerun **`python -m src.embeddings`**.
