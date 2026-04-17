# Raw Data

Inputs for `run_pipeline.py` go in **`data/raw/`**. CSVs are **gitignored**; the **CDTA shapefile** under `nyc_boundaries/` is **in the repo** so boundaries work without a separate download.

## Expected files (match `run_pipeline.py`)

Use these **exact names** under `data/raw/` (or change paths in `run_pipeline.py`):

| File | Role |
|------|------|
| `Bi-Annual_Pedestrian_Counts.csv` | Pedestrian count sensors (points) |
| `MTA_Subway_Stations_20260329.csv` | Subway stations (points); date in the name may change if you refresh the export |
| `DOHMH_New_York_City_Restaurant_Inspection_Results_20260329.csv` | Restaurant inspections → restaurant POI locations |
| `legally-operating-businesses-nyc.csv` | Active business licenses → retail / non-restaurant POI locations |
| `Public - Neighborhood Profiles 2018 - All.csv` | MOCEJ-style **Neighborhood Profiles** (community district statistics) |
| `Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool_20260415.csv` | **Optional but recommended:** NYC **Neighborhood Financial Health** tool export (merged into `nbhd_clean` as `nfh_*` columns when the file exists) |
| `nyc_boundaries/nycdta2020.shp` (+ `.shx`, `.dbf`, `.prj`, …) | **CDTA 2020** polygons for spatial join and area features |

The NFH CSV uses `CD` values such as `BX Community District 8`; the pipeline normalizes these to the same codes used for CDTA joins (e.g. `BX08`).

## Source datasets (links)

Upstream providers for the files in the table above. Rename downloads to match `run_pipeline.py` or edit paths there.

| Topic | What it feeds | Link |
|-------|----------------|------|
| Commerce | Restaurant / food POI (inspections) | [DOHMH Restaurant Inspection Results](https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j/about_data) |
| Commerce | Retail & licensed businesses | [NYC Issued Licenses](https://data.cityofnewyork.us/Business/Issued-Licenses/w7w3-xahh/about_data) |
| Commerce | Optional: businesses extract | [Kaggle — legally operating businesses](https://www.kaggle.com/datasets/new-york-city/ny-legally-operating-businesses) |
| Transit | Subway stations | [MTA Subway Stations (NY)](https://data.ny.gov/Transportation/MTA-Subway-Stations/39hk-dx4f/about_data) |
| Streets | Pedestrian counts | [NYC Bi-Annual Pedestrian Counts](https://data.cityofnewyork.us/Transportation/Bi-Annual-Pedestrian-Counts/cqsj-cfgu/about_data) |
| Profiles | ACS neighborhood economic stats | [NYC Comptroller — Neighborhood Economic Profiles](https://comptroller.nyc.gov/reports/nyc-neighborhood-economic-profiles/) |
| Boundaries | CDTA 2020 polygons | [NYC Planning — CDTA](https://www.nyc.gov/content/planning/pages/resources/datasets/community-district-tabulation) |
| Optional | Financial health (`nfh_*`) | [NYC Open Data — Neighborhood Financial Health](https://data.cityofnewyork.us/Business/Neighborhood-Financial-Health-Digital-Mapping-and-/r3dx-pew9/about_data) |

## Where to get files

1. **Hugging Face** (e.g. `ringoch/nyc-commercial-data`): download CSVs into `data/raw/` with names from the table; put all `nycdta2020.*` files under `data/raw/nyc_boundaries/` (Hub bundles often leave them in the archive root—move them).

2. **Manual:** pull the same datasets from NYC Open Data / DCP / Comptroller (see links above), rename to match `run_pipeline.py`.

3. **Processed `data/processed/`:** run `python run_pipeline.py` locally, or copy a pre-built folder from a teammate or release zip.

This repo is **code + placeholders**; raw CSVs are supplied separately unless you use a bundled dataset.
