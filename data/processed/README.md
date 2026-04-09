# Processed Data

This folder contains cleaned and engineered datasets prepared for downstream analysis, embedding, clustering, and modeling.

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
Cleaned area-level socioeconomic dataset derived from NYC Public Neighborhood Profiles.

**Main fields**
- `neighborhood`
- `cd` (community district)
- `borough`
- `construction_jobs`
- `manufacturing_jobs`
- `wholesale_jobs`
- `food_services`
- `total_businesses`
- `median_household_income`
- `commute_public_transit`
- `pct_bachelors_plus`

**Purpose**
- Represents neighborhood-level socioeconomic and business context  
- Supports aggregation, clustering, and neighborhood profiling  

---

## Final Feature Table

### `neighborhood_features_final.csv`

`neighborhood_features_final.csv` is the main output of the data processing and feature engineering pipeline. It is constructed using NYC CDTA (Community District Tabulation Area) boundaries, where each row represents a spatial unit.

- 71 CDTA areas  
- 22 engineered features  

This dataset integrates multiple data sources, including POI data, pedestrian counts, subway stations, and business activity indicators.

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
- `commercial_activity_score` (POI × pedestrian intensity)
- `transit_activity_score` (subway × pedestrian intensity)

---

### Data Integration Process

The dataset is constructed through the following steps:

1. Raw datasets are cleaned and standardized (`data_processing.py`)  
2. Point-level datasets (POI, pedestrian, subway) are spatially joined to CDTA boundaries  
3. Features are aggregated at the neighborhood level (`feature_engineering.py`)  
4. Missing values are handled using:
   - Mean imputation for pedestrian features (limited sensor coverage)  
   - Zero imputation for POI, retail, and subway features (interpreted as absence)  

---

### Notes

- The dataset is based on CDTA geography, not the original Neighborhood Profiles definitions  
- Some features (especially pedestrian activity) reflect data availability rather than full coverage  
- The table is designed for downstream tasks such as clustering, ranking, and predictive modeling  
