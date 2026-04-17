# Processed Data

This folder contains cleaned and engineered datasets prepared for downstream analysis, embedding, clustering, and the Streamlit app.

---

## Files

### `poi_clean.csv`

Cleaned point-level POI dataset combining multiple sources:

- Restaurant POIs from NYC restaurant inspection data  
- Retail POIs from legally operating business license data  

**Main fields**

- `business_name`
- `borough`
- `category`
- `latitude`
- `longitude`
- `poi_type`
- `description`

**Purpose**

- Represents commercial supply across NYC  
- Supports semantic embeddings, clustering, and location analysis  

---

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

### `nbhd_clean.csv`

Cleaned area-level socioeconomic dataset derived from **NYC Public Neighborhood Profiles** (MOCEJ-style column selection in `src/data_processing.py`), optionally merged with **Neighborhood Financial Health (NFH)** indicators when `data/raw/Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool_*.csv` is present and readable.

**Main fields (non-exhaustive)**

- `neighborhood`, `cd` (community district string, including combined districts such as `BX01 & BX02`), `borough`
- Job and business counts: `construction_jobs`, `manufacturing_jobs`, `wholesale_jobs`, `food_services`, `total_businesses`
- Demographics / SES proxies: `median_household_income`, `commute_public_transit` (**percent of 2016 employed** who commute via public transit — derived in `clean_neighborhood_profiles` from MOCEJ counts ÷ `2016 Employed`), `pct_bachelors_plus`, race population columns and derived `pct_*`, `total_jobs`, `total_population_proxy`
- **NFH (when merged):** `nfh_overall_score`, `nfh_goal4_fin_shocks_score`, other `nfh_goal*_score` / `*_rank`, and demographic columns such as `nfh_median_income`, `nfh_poverty_rate`, `nfh_pct_*` (see CSV header for the full list)

**Purpose**

- Joined onto the CDTA master table by a normalized district key (`normalize_cdta_join_key` in `src/feature_engineering.py`). NFH source rows use labels like `BX Community District 8`; these are normalized to codes such as `BX08` in `src/data_processing.py` so they align with the same join logic as MOCEJ rows.

---

## Final Feature Table

### `neighborhood_features_final.csv`

- Written once by **`run_feature_engineering()`** in `src/feature_engineering.py` (also invoked from **`run_pipeline.py`**). This is the only merged feature table **`app.py`**, **`pages/Ranking.py`**, and **`src/embeddings.py`** read.

Each row is one **CDTA** (Community District Tabulation Area) polygon from `nycdta2020`. With MOCEJ profiles plus NFH merged, a typical build is **on the order of ~71 rows** (exact column count depends on optional NFH columns and pipeline version).

**Not in this table:** There is **no** `persistence_score` column in the current pipeline; the app blends **semantic similarity** and **`commercial_activity_score`** only (see below).

Sources integrated:

- POI (restaurant + retail licenses), pedestrian counts, subway stations (aggregated after spatial join)  
- CDTA area (`area_km2`) and density / interaction features  
- **MOCEJ + NFH columns** from `nbhd_clean`, merged on the normalized district key. Where a CDTA still has no direct match after normalization, **numeric** profile and `nfh_*` columns are **imputed** in `merge_all_features` (borough median, then citywide median) so the final CSV has **no NaNs** and the Streamlit hard filters stay usable. Treat imputed values as **proxies**, not tract-level ground truth.

---

### Feature Groups

**Commercial Activity (POI-based)**

- `total_poi`, `unique_poi`
- `category_diversity`, `category_entropy`
- `poi_density_per_km2`

**Retail and Category Structure**

- **`food`**, **`retail`**, **`other`**: per-CDTA counts from `simplify_category` in `src/feature_engineering.py` (grounded in `poi_type`: DOHMH = restaurant, licenses = retail). Under current rules these align with former `num_restaurant` / `num_retail` counts, so **only the simplified columns are exported** (no duplicate `num_*` columns).
- **`ratio_restaurant`** (= `food` / `total_poi`) and **`ratio_retail`** (= `retail` / `total_poi`); **`food_to_retail_ratio`**
- **`retail_density_per_km2`**, **`food_density_per_km2`** (per km² of CDTA area)

**Pedestrian Activity**

- `avg_pedestrian`, `peak_pedestrian`
- `pedestrian_count_points`

**Transit Accessibility**

- `subway_station_count`
- `subway_density_per_km2`

**Interaction Features**

- `commercial_activity_score` = **`total_poi` × `avg_pedestrian`** (after missing POI counts are set to 0 and pedestrian averages are filled with the citywide mean, then any remaining NaN pedestrian with 0). Used in **`app.py`** after MinMax scaling alongside semantic similarity.  
- `transit_activity_score` = **`subway_station_count` × `avg_pedestrian`** with the same pedestrian handling. A value can still be **0** when `total_poi` or `subway_station_count` is 0 or pedestrian signal is 0.

**Geometry / area**

- `area_km2`

**Neighborhood profile (MOCEJ) and NFH**

Examples: `median_household_income`, `pct_bachelors_plus`, `commute_public_transit`, `total_businesses`, job counts, race population proxies; `nfh_overall_score`, `nfh_goal4_fin_shocks_score`, and other `nfh_*` columns when the NFH CSV was merged in `data_processing`. See the CSV header for the full list.

---

## Downstream: embeddings and Streamlit

1. **`python -m src.embeddings`** reads **`neighborhood_features_final.csv`**, builds one text profile per row (`src/embeddings.py`), calls OpenAI **`text-embedding-3-small`**, and saves **`outputs/embeddings/neighborhood_embeddings.npy`** and **`neighborhood_texts.npy`**.  
2. **`streamlit run app.py`** — home is **K-Selection** (`app.py`); open **Ranking** (`pages/Ranking.py`) for filters and blend. Both load the CSV (cached) and embeddings (cached).  
3. **Ranking page — hard filters:** DuckDB `SELECT` with sidebar thresholds (borough, subway, pedestrian, POI density, total POI, **`commercial_activity_score`**, etc.).  
4. **Soft ranking:** On the filtered rows, **MinMaxScaler** is fit on **`[cosine_sim, commercial_activity_score]`**. One blend slider **α**; **β = 1 − α**. Output: **`blended_score`**. Optional **CDTA map** (choropleth by `blended_score`) when embeddings and the shapefile are available.  
5. Optional **NFH thresholds** when those columns exist.  
6. Optional **Claude** panel: read-only SQL on the filtered dataframe (`ANTHROPIC_API_KEY`).

Details and setup: **root `README.md`**.

---

### Data Integration Process

1. Raw datasets are cleaned (`src/data_processing.py`) → `*_clean.csv` under `data/processed/`.  
2. Point layers are spatially joined to CDTA boundaries; features are aggregated per CDTA (`src/feature_engineering.py`).  
3. **`nbhd_clean` is merged** onto the CDTA master table using a normalized district code (see `normalize_cdta_join_key` in `src/feature_engineering.py`).  
4. **Missing values (spatial / activity columns):** POI / retail / subway counts and densities are filled with **0** where appropriate. **Pedestrian:** `avg_pedestrian` / `peak_pedestrian` use **mean** then **0** for any residual NaN; `pedestrian_count_points` missing → **0**.  
5. **Interaction scores** (`commercial_activity_score`, `transit_activity_score`) are computed **after** those fills so they reflect imputed inputs, not premature `× 0` from NaNs.  
6. **Profile + `nfh_*` numeric columns** after the merge: **borough median**, then **citywide median**, for any remaining NaN.  
7. **`run_pipeline.py`** prints a NaN summary; a **clean** build should report **no missing values** in the final feature table.

---

### Notes

- **Geography** is CDTA; neighborhood-profile and NFH data enter as **attributes**, not as the boundary source.  
- **Combined community districts** (e.g. `BX01 & BX02` in MOCEJ, or `BX Community Districts 1 & 2` in NFH) are normalized to a **single** code (first district number) for joining; that is an intentional mapping, not duplicate rows in the source.  
- Pedestrian coverage is limited by where sensors exist.  
- This folder’s CSVs are **committed in the repo** for convenience; regenerate anytime with **`python run_pipeline.py`** after changing code or raw inputs.  
- If OpenAI returns **429 / insufficient_quota** when embedding, fix billing or quota for `OPENAI_API_KEY`, then rerun **`python -m src.embeddings`**.
