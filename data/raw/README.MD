# Raw Data

Source files for the pipeline live under `data/raw/`. They are **not committed to GitHub** (see repo `.gitignore`) so the clone stays small.

## Where to get files (e.g. live demo on a new laptop)

1. **Hugging Face dataset (if your team publishes one)**  
   Example: **`ringoch/nyc-commercial-data`**. On the Hub, CSVs and the CDTA shapefile pieces (`nycdta2020.shp`, `.shx`, `.dbf`, `.prj`, `.shp.xml`) usually sit in the **repo root** together — not inside a `nyc_boundaries/` folder. After you download:

   - Put all **`.csv`** files under **`data/raw/`** using the **same names as on Hugging Face** (see `run_pipeline.py`).
   - Create **`data/raw/nyc_boundaries/`** and **move** every file named **`nycdta2020.*`** into that folder. The pipeline expects **`data/raw/nyc_boundaries/nycdta2020.shp`** (GeoPandas reads the sidecars next to it).

2. **NYC Open Data / DCP**  
   Download the same datasets manually and place or rename them to match `run_pipeline.py` (pedestrian counts, subway stations, restaurant inspections, business licenses, Neighborhood Profiles CSV, and **CDTA 2020** shapefile under `data/raw/nyc_boundaries/`).

3. **Processed outputs (`data/processed/`)**  
   These are also **ignored by Git** by default. For a demo you either:
   - **Generate locally:** install dependencies (`requirements.txt`, includes `geopandas`), put raw data in place, then run  
     `python run_pipeline.py`  
   - **Copy a pre-built folder** from a teammate, USB, or a shared release zip / Hugging Face file bundle — not from the GitHub tree unless someone force-committed artifacts (not the default workflow).

The GitHub repo ships **code + placeholders** (`.gitkeep`); **data must be supplied separately** unless your team documents another channel.
