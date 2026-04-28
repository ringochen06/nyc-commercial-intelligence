// Mirrors api/schemas.py. Keep in sync.

export type Vintage = "present";

export interface FeatureRange {
  min: number;
  max: number;
}

export interface FeatureRangesResponse {
  boroughs: string[];
  ranges: Record<string, FeatureRange>;
  has_nfh_goal4: boolean;
  has_nfh_overall: boolean;
  activity_columns: string[];
  density_columns: string[];
}

export interface ClusterPoint {
  neighborhood: string;
  cd: string | null;
  borough: string | null;
  map_key: string | null;
  cluster: number;
  raw: Record<string, number>;
}

export interface ClusterSummary {
  cluster: number;
  size: number;
  description: string;
  centroid_z: number[];
}

export interface ClusterResponse {
  k_range: number[];
  inertias: number[];
  silhouettes_numpy: number[];
  silhouettes_sklearn: number[];
  elbow_k: number;
  elbow_k_kneedle: number;
  best_silhouette_k: number;
  chosen_k: number;
  features: string[];
  feature_means: number[];
  feature_stds: number[];
  points: ClusterPoint[];
  centroids_z: number[][];
  cluster_summaries: ClusterSummary[];
}

export interface HardFilters {
  boroughs?: string[];
  min_subway_stations?: number;
  min_avg_pedestrian?: number;
  min_storefront_density?: number;
  min_storefront_filings?: number;
  min_commercial_activity?: number;
  max_competitive_score?: number;
  max_shooting_incident_count?: number;
  min_nfh_goal4?: number;
  min_nfh_overall?: number;
}

export interface RankRequest {
  query: string;
  alpha: number;
  filters: HardFilters;
  vintage: Vintage;
  competitive_source?: string;
  cluster_assignments?: Record<string, number>;
  cluster_briefs?: Record<string, string>;
}

export interface RankRow {
  rank: number;
  neighborhood: string;
  cd: string | null;
  borough: string | null;
  map_key: string | null;
  semantic_similarity: number;
  specific_competitive_score: number;
  blended_score: number;
  cluster: number | null;
  cluster_description: string | null;
}

export interface RankResponse {
  rows: RankRow[];
  n_total: number;
  n_filtered: number;
  sql: string;
}

export interface CdtaGeoResponse {
  geojson: GeoJSON.FeatureCollection;
  bounds: { minx: number; miny: number; maxx: number; maxy: number };
  center: { lat: number; lon: number };
}

export interface HealthResponse {
  status: string;
  has_cdta_shapefile: boolean;
  has_anthropic_key: boolean;
  has_openai_key: boolean;
}
