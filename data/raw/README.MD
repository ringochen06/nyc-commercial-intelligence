# Raw Data

Source files for the pipeline live under `data/raw/`. **CSVs are not committed** (see `.gitignore`); the **CDTA `nycdta2020.*` shapefile** under `nyc_boundaries/` **is committed** in this repo so you can run the pipeline without downloading boundaries.

## Expected files (match `run_pipeline.py`)

Place these **exact filenames** under `data/raw/` (or update the paths in `run_pipeline.py` if your downloads use different names):

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

## Where to get files (e.g. live demo on a new laptop)

1. **Hugging Face dataset (if your team publishes one)**  
   Example: **`ringoch/nyc-commercial-data`**. On the Hub, CSVs and the CDTA shapefile pieces (`nycdta2020.shp`, `.shx`, `.dbf`, `.prj`, `.shp.xml`) usually sit in the **repo root** together — not inside a `nyc_boundaries/` folder. After you download:

   - Put all **`.csv`** files under **`data/raw/`** using the **same names as on Hugging Face** (see table above and `run_pipeline.py`).
   - Create **`data/raw/nyc_boundaries/`** and **move** every file named **`nycdta2020.*`** into that folder. The pipeline expects **`data/raw/nyc_boundaries/nycdta2020.shp`** (GeoPandas reads the sidecars next to it).

2. **NYC Open Data / DCP**  
   Download the same datasets manually and place or rename them to match `run_pipeline.py` (pedestrian counts, subway stations, restaurant inspections, business licenses, Neighborhood Profiles CSV, **Neighborhood Financial Health** export if you want NFH columns, and **CDTA 2020** shapefile under `data/raw/nyc_boundaries/`).

3. **Processed outputs (`data/processed/`)**  
   These are also **ignored by Git** by default. For a demo you either:
   - **Generate locally:** install dependencies (`requirements.txt`, includes `geopandas`), put raw data in place, then run  
     `python run_pipeline.py`  
   - **Copy a pre-built folder** from a teammate, USB, or a shared release zip / Hugging Face file bundle — not from the GitHub tree unless someone force-committed artifacts (not the default workflow).

The GitHub repo ships **code + placeholders** (`.gitkeep`); **data must be supplied separately** unless your team documents another channel.
