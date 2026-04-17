from src.data_processing import run_data_processing
from src.feature_engineering import run_feature_engineering

print("Step 1: running data processing...")
processed = run_data_processing(
    pedestrian_path="data/raw/Bi-Annual_Pedestrian_Counts.csv",
    subway_path="data/raw/MTA_Subway_Stations_20260329.csv",
    restaurant_path="data/raw/DOHMH_New_York_City_Restaurant_Inspection_Results_20260329.csv",
    license_path="data/raw/legally-operating-businesses-nyc.csv",
    nbhd_path="data/raw/Public - Neighborhood Profiles 2018 - All.csv",
    nfh_path="data/raw/Neighborhood_Financial_Health_Digital_Mapping_and_Data_Tool_20260415.csv",
    output_dir="data/processed"
)

print("Data processing finished.")
print("POI shape:", processed["poi"].shape)
print("Subway shape:", processed["subway"].shape)
print("Pedestrian shape:", processed["pedestrian"].shape)
print("Neighborhood shape:", processed["neighborhood"].shape)

print("\nStep 2: running feature engineering...")

outputs = run_feature_engineering(
    poi_path="data/processed/poi_clean.csv",
    pedestrian_path="data/processed/ped_clean.csv",
    subway_path="data/processed/subway_clean.csv",
    nbhd_clean_path="data/processed/nbhd_clean.csv",
    boundary_path="data/raw/nyc_boundaries/nycdta2020.shp",
    output_dir="data/processed"

)
print(f"poi output: {outputs.get("poi_features").columns.tolist()}")

# `neighborhood_features_final.csv` is written inside `run_feature_engineering`.
df_final = outputs["neighborhood_features"]
df_final.to_csv("data/processed/neighborhood_features_final.csv", index=False)

print("Feature engineering finished.")
print("Final feature table shape:", df_final.shape)
na_counts = df_final.isna().sum()
na_cols = na_counts[na_counts > 0].sort_values(ascending=False)
if len(na_cols):
    print("\nNaN counts (investigate before shipping):")
    print(na_cols.head(20))
else:
    print("\nNo missing values in the final table.")

print("\nPreview:")
print(df_final.head())