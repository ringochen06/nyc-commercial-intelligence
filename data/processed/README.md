# Processed Data

This folder contains cleaned datasets prepared for downstream analysis, embedding, clustering, and modeling.

## Files

### `poi_clean.csv`
Cleaned POI dataset combining:
- point-level (each row represents a business location)
- restaurant POIs from NYC restaurant inspection data
- retail POIs from legally operating business license data

Main fields:
- `business_name`
- `borough`
- `category`
- `latitude`
- `longitude`
- `poi_type`
- `description`

Purpose:
- represents commercial supply across NYC
- supports semantic embeddings, clustering, and location analysis

---

### `ped_clean.csv`
Cleaned pedestrian count dataset.
- point-level (pedestrian count locations)

Main fields:
- `borough`
- `street`
- `latitude`
- `longitude`
- `avg_pedestrian`
- `peak_pedestrian`

Purpose:
- represents pedestrian demand / foot traffic
- supports downstream demand estimation and spatial analysis

---

### `subway_clean.csv`
Cleaned subway station dataset.
- point-level (subway stations)

Main fields:
- `station_name`
- `borough`
- `latitude`
- `longitude`
- `routes`

Purpose:
- represents transit accessibility
- supports future distance-to-subway or accessibility features

---

### `nbhd_clean.csv`
Cleaned neighborhood-level socioeconomic dataset derived from NYC Public Neighborhood Profiles.
- area-level (community district)

Main fields include:
- `neighborhood`
- `cd`
- `borough`
- `construction_jobs`
- `manufacturing_jobs`
- `wholesale_jobs`
- `food_services`
- `total_businesses`
- `median_household_income`
- `commute_public_transit`
- `pct_bachelors_plus`

Purpose:
- represents area-level socioeconomic and business context
- supports future aggregation, clustering, and neighborhood profiling

## Notes
- Datasets are intentionally kept separate for flexibility.
- No final spatial merge is applied at this stage.
- Borough names and coordinate fields were standardized where possible.
