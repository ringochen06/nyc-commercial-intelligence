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

Cleaned area-level socioeconomic dataset derived from NYC Public Neighborhood Profiles (column selection in `src/data_processing.py`).

**Main fields (non-exhaustive)**

- `neighborhood`, `cd` (community district string), `borough`
- Job and business counts: `construction_jobs`, `manufacturing_jobs`, `wholesale_jobs`, `food_services`, `total_businesses`
- Demographics / SES proxies: `median_household_income`, `commute_public_transit`, `pct_bachelors_plus`, race share columns and derived `pct_*` fields

**Purpose**

- Joined to the CDTA-level feature table by normalized community-district / CDTA codes (`src/feature_engineering.py`), not used as the spatial base map. Row count is typically **smaller** than the number of CDTA polygons (e.g. ~55 profile rows vs ~71 CDTAs), so **many** CDTA rows get profile columns; **some** CDTAs have **no** matching profile row and keep **NaN** for those attributes.

---

## Final Feature Table

### `neighborhood_features.csv` and `neighborhood_features_final.csv`

- **`neighborhood_features.csv`** — written by `run_feature_engineering()` in `src/feature_engineering.py`.
- **`neighborhood_features_final.csv`** — same table, saved again by `run_pipeline.py` as the default input for **`app.py`** and **`src/embeddings.py`**.

Each row is one **CDTA** (Community District Tabulation Area) polygon from `nycdta2020`. Typical size with the current pipeline: **on the order of ~71 rows × ~38 columns** (exact counts depend on boundary + profile join).

**Not in this table:** There is **no** `persistence_score` column in the current pipeline; the app blends **semantic similarity** and **`commercial_activity_score`** only (see below).

Sources integrated:

- POI (restaurant + retail licenses), pedestrian counts, subway stations (aggregated after spatial join)  
- CDTA area (`area_km2`) and density / interaction features  
- **Neighborhood profile columns** from `nbhd_clean` merged where `cd` keys match after normalization (some CDTA rows may have no matching profile row — **expected**, not bad raw data)

---

### Feature Groups

**Commercial Activity (POI-based)**

- `total_poi`, `unique_poi`
- `category_diversity`, `category_entropy`
- `poi_density_per_km2`

**Retail and Category Structure**

- `num_retail`, `retail`, `food`, `other`
- `ratio_retail`, `retail_density_per_km2`

**Pedestrian Activity**

- `avg_pedestrian`, `peak_pedestrian`
- `pedestrian_count_points`

**Transit Accessibility**

- `subway_station_count`
- `subway_density_per_km2`

**Interaction Features**

- `commercial_activity_score` (POI × pedestrian intensity) — used in **`app.py`** after MinMax scaling alongside semantic similarity  
- `transit_activity_score` (subway × pedestrian intensity)

**Geometry / area**

- `area_km2`

**Neighborhood profile (when join succeeds)**  

Examples: `median_household_income`, `pct_bachelors_plus`, `commute_public_transit`, `total_businesses`, job counts, race population proxies — see column names in the CSV for the full list.

---

## Downstream: embeddings and Streamlit (`app.py`)

1. **`python -m src.embeddings`** reads **`neighborhood_features_final.csv`**, builds one text profile per row (`src/embeddings.py`), calls OpenAI **`text-embedding-3-small`**, and saves **`outputs/embeddings/neighborhood_embeddings.npy`** and **`neighborhood_texts.npy`**.  
2. **`streamlit run app.py`** loads the CSV (cached) and embeddings (cached).  
3. **Hard filters:** DuckDB `SELECT` with sidebar thresholds (borough, subway, pedestrian, POI density, total POI, **`commercial_activity_score`**, etc.).  
4. **Soft ranking:** On the filtered rows, **MinMaxScaler** is fit on **`[cosine_sim, commercial_activity_score]`**. The user sets **one** blend slider **α** for semantic weight; **β = 1 − α** for the scaled activity column. Output column: **`blended_score`**.  
5. Optional **Claude** panel: read-only SQL on the filtered dataframe (`ANTHROPIC_API_KEY`).

Details and setup: **root `README.md`**.

---

### Data Integration Process

1. Raw datasets are cleaned (`src/data_processing.py`) → `*_clean.csv` under `data/processed/`.  
2. Point layers are spatially joined to CDTA boundaries; features are aggregated per CDTA (`src/feature_engineering.py`).  
3. **`nbhd_clean` is merged** onto the CDTA master table using a normalized district code (see `normalize_cdta_join_key`).  
4. **Missing values (spatial / activity columns):**
   - Pedestrian aggregates: mean imputation for `avg_pedestrian` / `peak_pedestrian` where missing after aggregation.  
   - POI / retail / subway-related counts and densities: filled with **0** where appropriate.  
5. **Profile columns:** may remain **NaN** for CDTA polygons with no matching neighborhood-profile row. `run_pipeline.py` prints NaN counts with a note that this is **expected** for join gaps, not necessarily corrupt sources.

---

### Notes

- **Geography** is CDTA; neighborhood-profile data enters as **attributes**, not as the boundary source.  
- Pedestrian coverage is limited by where sensors exist.  
- This folder’s CSVs are **generated locally** and are **gitignored** by default; cloning the repo alone does not include them.  
- If OpenAI returns **429 / insufficient_quota** when embedding, fix billing or quota for `OPENAI_API_KEY`, then rerun **`python -m src.embeddings`**.
