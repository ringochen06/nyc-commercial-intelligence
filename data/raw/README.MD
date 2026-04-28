# Raw Data

Inputs for `run_pipeline.py` go in **`data/raw/`**. CSVs are **gitignored**; the **CDTA shapefile** under `nyc_boundaries/` is **in the repo** so boundaries work without a separate download.

## Expected files (match `run_pipeline.py`)

Use these **exact names** under `data/raw/` (or change paths in `run_pipeline.py`):

| File | Role |
|------|------|
| `Bi-Annual_Pedestrian_Counts.csv` | DOT **bi-annual pedestrian counts**: sensor locations (points) with time-of-day columns; cleaned in step 1, then aggregated per **CDTA** in step 2 for foot-traffic features (`avg_pedestrian`, `peak_pedestrian`, etc.). |
| `MTA_Subway_Stations_20260329.csv` | **Subway station** point locations (entrances/coordinates); the date in the filename may change when you re-export. Aggregated per **CDTA** in step 2 for station counts and transit density. |
| `Public - Neighborhood Profiles 2018 - All.csv` | **MOCEJ Public Neighborhood Profiles**: one row per community district—jobs, demographics, income, education, commute share, etc. Cleaned to **`nbhd_clean.csv`** in step 1 and merged onto the CDTA feature table in step 2. |
| `Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool_20260415.csv` | **Optional (recommended):** NYC **Neighborhood Financial Health** export. When present, its indicators are merged into **`nbhd_clean.csv`** as **`nfh_*`** columns; if the file is missing, the pipeline still runs without NFH fields. |
| `Storefronts_Reported_Vacant_or_Not_20260424.csv` | **Optional:** **Storefronts Reported Vacant or Not** point filings (vacancy and primary-business-activity fields from the city export). Read **only in step 2** (`run_feature_engineering`); not copied into `data/processed/`. The date in the filename may change when you refresh the export. If the file is absent, storefront-related columns in **`neighborhood_features_final.csv`** are zero-filled. |
| `NYPD Shooting Incident Data2024.csv` | **NYPD Shooting Incident Data2024** (historic export). Point incidents with coordinates; can be spatially aggregated to **CDTA / neighborhood** totals and merged into `data/processed/neighborhood_features_final.csv` as `shooting_incident_count_2024`. |
| `nyc_boundaries/nycdta2020.shp` (+ `.shx`, `.dbf`, `.prj`, …) | **CDTA 2020** boundary polygons (DCP): defines each neighborhood polygon for spatial joins, **`area_km2`**, and the master row list. Keep the full shapefile bundle, not only `.shp`. |

The NFH CSV uses `CD` values such as `BX Community District 8`; the pipeline normalizes these to the same codes used for CDTA joins (e.g. `BX08`).

## Source datasets (links)

Upstream providers for the files in the table above. Rename downloads to match `run_pipeline.py` or edit paths there.

| Topic | What it feeds | Link |
|-------|----------------|------|
| Transit | Subway stations | [MTA Subway Stations (NY)](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f/about_data) |
| Streets | Pedestrian counts | [NYC Bi-Annual Pedestrian Counts](https://data.cityofnewyork.us/Transportation/Bi-Annual-Pedestrian-Counts/cqsj-cfgu/about_data) |
| Profiles | ACS neighborhood economic stats | [NYC Comptroller — Neighborhood Economic Profiles](https://comptroller.nyc.gov/reports/nyc-neighborhood-economic-profiles/) |
| Boundaries | CDTA 2020 polygons | [NYC Planning — CDTA](https://www.nyc.gov/content/planning/pages/resources/datasets/community-district-tabulation) |
| Optional | Financial health (`nfh_*`) | [NYC Open Data — Neighborhood Financial Health](https://data.cityofnewyork.us/Business/Neighborhood-Financial-Health-Digital-Mapping-and-/r3dx-pew9/about_data) |
| Vacancy filings | Storefront reports → `storefront_*` CDTA features in `neighborhood_features_final.csv` | [NYC Open Data — Storefronts Reported Vacant or Not](https://data.cityofnewyork.us/City-Government/Storefronts-Reported-Vacant-or-Not/92iy-9c3n/about_data) |
| Public safety | Shooting incidents (optional CDTA aggregate) | [Kaggle — NYPD Shooting Incident Data Historic](https://www.kaggle.com/datasets/anthonyylee/nypd-shooting-incident-data-historic) |

## Where to get files

1. **Hugging Face** (e.g. `ringoch/nyc-commercial-data`): download CSVs into `data/raw/` with names from the table; put all `nycdta2020.*` files under `data/raw/nyc_boundaries/` (Hub bundles often leave them in the archive root—move them).

2. **Manual:** pull the same datasets from NYC Open Data / DCP / Comptroller (see links above), rename to match `run_pipeline.py`.

3. **Processed `data/processed/`:** run `python run_pipeline.py` locally, or copy a pre-built folder from a teammate or release zip.

This repo is **code + placeholders**; raw CSVs are supplied separately unless you use a bundled dataset.
