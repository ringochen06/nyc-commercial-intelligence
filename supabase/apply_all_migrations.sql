-- Concatenated Supabase migrations 0001-0009, in order.
-- Paste into Supabase Dashboard -> SQL Editor -> Run (one shot).
-- Safe to run on a fresh project; creates extension, table, indexes, RLS, RPC.


-- ============================================================
-- 0001_extensions.sql
-- ============================================================

-- pgvector for embedding similarity search.
-- Supabase already exposes the extension; this just enables it in the public schema.
create extension if not exists vector;


-- ============================================================
-- 0002_neighborhoods.sql
-- ============================================================

-- Master neighborhood feature table.
-- One row per CDTA (Community District Tabulation Area), mirroring
-- data/processed/neighborhood_features_final.csv plus a Voyage AI
-- voyage-4 (1024-dim) vector built from the row's text profile.

create table public.neighborhoods (
    neighborhood                                text primary key,
    cd                                          text not null unique,
    borough                                     text not null,

    area_km2                                    double precision,
    avg_pedestrian                              double precision,
    peak_pedestrian                             double precision,
    pedestrian_count_points                     integer,
    subway_station_count                        integer,
    storefront_filing_count                     integer,

    act_accounting_services_storefront          integer,
    act_broadcasting_telecomm_storefront        integer,
    act_educational_services_storefront         integer,
    act_finance_and_insurance_storefront        integer,
    act_food_services_storefront                integer,
    act_health_care_or_social_assistance_storefront integer,
    act_information_services_storefront         integer,
    act_legal_services_storefront               integer,
    act_manufacturing_storefront                integer,
    act_movies_video_sound_storefront           integer,
    act_no_business_activity_identified_storefront integer,
    act_other_storefront                        integer,
    act_publishing_storefront                   integer,
    act_real_estate_storefront                  integer,
    act_retail_storefront                       integer,
    act_unknown_storefront                      integer,
    act_wholesale_storefront                    integer,
    act_other_lower_storefront                  integer,

    construction_jobs                           double precision,
    manufacturing_jobs                          double precision,
    wholesale_jobs                              double precision,
    pop_black                                   double precision,
    pop_hispanic                                double precision,
    pop_asian                                   double precision,
    total_population_proxy                      double precision,
    food_services                               double precision,
    total_businesses                            double precision,
    commute_public_transit                      double precision,
    pct_bachelors_plus                          double precision,
    total_jobs                                  double precision,

    nfh_median_income                           double precision,
    nfh_poverty_rate                            double precision,
    nfh_pct_white                               double precision,
    nfh_pct_black                               double precision,
    nfh_pct_asian                               double precision,
    nfh_pct_hispanic                            double precision,
    nfh_goal1_fin_services_score                double precision,
    nfh_goal2_goods_services_score              double precision,
    nfh_goal3_jobs_income_score                 double precision,
    nfh_goal4_fin_shocks_score                  double precision,
    nfh_goal5_build_assets_score                double precision,
    nfh_overall_score                           double precision,

    category_diversity                          double precision,
    category_entropy                            double precision,

    act_accounting_services_density             double precision,
    act_broadcasting_telecomm_density           double precision,
    act_educational_services_density            double precision,
    act_finance_and_insurance_density           double precision,
    act_food_services_density                   double precision,
    act_health_care_or_social_assistance_density double precision,
    act_information_services_density            double precision,
    act_legal_services_density                  double precision,
    act_manufacturing_density                   double precision,
    act_movies_video_sound_density              double precision,
    act_no_business_activity_identified_density double precision,
    act_other_density                           double precision,
    act_publishing_density                      double precision,
    act_real_estate_density                     double precision,
    act_retail_density                          double precision,
    act_unknown_density                         double precision,
    act_wholesale_density                       double precision,
    act_other_lower_density                     double precision,

    subway_density_per_km2                      double precision,
    storefront_density_per_km2                  double precision,
    commercial_activity_score                   double precision,
    transit_activity_score                      double precision,

    embedding                                   vector(1024),
    embedding_text                              text,

    pipeline_loaded_at                          timestamptz not null default now()
);

-- The CSV has two columns that differ only in case: act_OTHER_storefront and
-- act_other_storefront. Postgres folds identifiers to lowercase, so the loader
-- maps the lowercase ("act_other") variant to *_lower_storefront / *_lower_density.

comment on table public.neighborhoods is
    'One row per NYC CDTA. Source: data/processed/neighborhood_features_final.csv plus Voyage AI voyage-4 embeddings of the row text profile.';
comment on column public.neighborhoods.cd is
    'CDTA 2020 code (e.g. MN01). Joins to nycdta2020 GeoJSON used for the choropleth.';
comment on column public.neighborhoods.embedding is
    'Voyage AI voyage-4 vector of embedding_text. Cosine distance via the <=> operator.';
comment on column public.neighborhoods.embedding_text is
    'Exact text profile used to produce embedding. Stored so we can rebuild vectors deterministically.';


-- ============================================================
-- 0003_indexes.sql
-- ============================================================

-- Borough is the most common hard filter; index it even though the table is small (~70 rows).
create index neighborhoods_borough_idx
    on public.neighborhoods (borough);

-- HNSW index for cosine similarity. At ~70 rows a sequential scan is actually fastest,
-- so this is mainly forward-looking; pgvector still uses it correctly when it helps.
-- Switch to ivfflat with `lists = sqrt(rowcount)` if the table grows past a few thousand rows.
create index neighborhoods_embedding_hnsw_idx
    on public.neighborhoods
    using hnsw (embedding vector_cosine_ops);


-- ============================================================
-- 0004_rls.sql
-- ============================================================

-- Row-level security: the public anon key is allowed to SELECT the neighborhoods table
-- (the dashboard is a read-only public app). All writes go through the service-role key
-- used by the loader script and are therefore exempt from RLS.

alter table public.neighborhoods enable row level security;

create policy "neighborhoods_anon_read"
    on public.neighborhoods
    for select
    to anon, authenticated
    using (true);


-- ============================================================
-- 0005_match_neighborhoods_rpc.sql
-- ============================================================

-- RPC for the Ranking page: take a query embedding plus an optional set of hard
-- filters, and return rows ordered by cosine similarity. Doing the cosine math in
-- Postgres avoids shipping all 1024-dim vectors over the wire to the Vercel function.

create or replace function public.match_neighborhoods(
    query_embedding             vector(1024),
    boroughs                    text[]            default null,
    min_subway_station_count    integer           default null,
    min_avg_pedestrian          double precision  default null,
    min_storefront_density      double precision  default null,
    min_storefront_filing_count integer           default null,
    min_commercial_activity     double precision  default null,
    min_nfh_overall_score       double precision  default null,
    min_nfh_goal4_score         double precision  default null,
    match_count                 integer           default 50
)
returns table (
    neighborhood                text,
    cd                          text,
    borough                     text,
    commercial_activity_score   double precision,
    transit_activity_score      double precision,
    avg_pedestrian              double precision,
    subway_station_count        integer,
    storefront_filing_count     integer,
    storefront_density_per_km2  double precision,
    nfh_overall_score           double precision,
    similarity                  double precision
)
language sql
stable
as $$
    select
        n.neighborhood,
        n.cd,
        n.borough,
        n.commercial_activity_score,
        n.transit_activity_score,
        n.avg_pedestrian,
        n.subway_station_count,
        n.storefront_filing_count,
        n.storefront_density_per_km2,
        n.nfh_overall_score,
        1 - (n.embedding <=> query_embedding) as similarity
    from public.neighborhoods n
    where n.embedding is not null
      and (boroughs is null or n.borough = any (boroughs))
      and (min_subway_station_count is null or n.subway_station_count >= min_subway_station_count)
      and (min_avg_pedestrian is null or n.avg_pedestrian >= min_avg_pedestrian)
      and (min_storefront_density is null or n.storefront_density_per_km2 >= min_storefront_density)
      and (min_storefront_filing_count is null or n.storefront_filing_count >= min_storefront_filing_count)
      and (min_commercial_activity is null or n.commercial_activity_score >= min_commercial_activity)
      and (min_nfh_overall_score is null or n.nfh_overall_score >= min_nfh_overall_score)
      and (min_nfh_goal4_score is null or n.nfh_goal4_fin_shocks_score >= min_nfh_goal4_score)
    order by n.embedding <=> query_embedding
    limit match_count;
$$;

grant execute on function public.match_neighborhoods(
    vector, text[], integer, double precision, double precision, integer,
    double precision, double precision, double precision, integer
) to anon, authenticated;


-- ============================================================
-- 0006_pipeline_refresh.sql
-- ============================================================

-- Refresh public.neighborhoods to match the slimmed pipeline output
-- (data/processed/neighborhood_features_final.csv as of 2026-04-29).
--
-- Adds three new pipeline columns and drops the per-activity industry breakdown
-- (act_*_storefront / act_*_density) and the NFH score family — those are no
-- longer produced by run_pipeline.py.

alter table public.neighborhoods
    add column if not exists shooting_incident_count   integer,
    add column if not exists median_household_income   double precision,
    add column if not exists competitive_score         double precision;

alter table public.neighborhoods
    drop column if exists act_accounting_services_storefront,
    drop column if exists act_broadcasting_telecomm_storefront,
    drop column if exists act_educational_services_storefront,
    drop column if exists act_finance_and_insurance_storefront,
    drop column if exists act_food_services_storefront,
    drop column if exists act_health_care_or_social_assistance_storefront,
    drop column if exists act_information_services_storefront,
    drop column if exists act_legal_services_storefront,
    drop column if exists act_manufacturing_storefront,
    drop column if exists act_movies_video_sound_storefront,
    drop column if exists act_no_business_activity_identified_storefront,
    drop column if exists act_other_storefront,
    drop column if exists act_publishing_storefront,
    drop column if exists act_real_estate_storefront,
    drop column if exists act_retail_storefront,
    drop column if exists act_unknown_storefront,
    drop column if exists act_wholesale_storefront,
    drop column if exists act_other_lower_storefront,
    drop column if exists act_accounting_services_density,
    drop column if exists act_broadcasting_telecomm_density,
    drop column if exists act_educational_services_density,
    drop column if exists act_finance_and_insurance_density,
    drop column if exists act_food_services_density,
    drop column if exists act_health_care_or_social_assistance_density,
    drop column if exists act_information_services_density,
    drop column if exists act_legal_services_density,
    drop column if exists act_manufacturing_density,
    drop column if exists act_movies_video_sound_density,
    drop column if exists act_no_business_activity_identified_density,
    drop column if exists act_other_density,
    drop column if exists act_publishing_density,
    drop column if exists act_real_estate_density,
    drop column if exists act_retail_density,
    drop column if exists act_unknown_density,
    drop column if exists act_wholesale_density,
    drop column if exists act_other_lower_density,
    drop column if exists nfh_median_income,
    drop column if exists nfh_poverty_rate,
    drop column if exists nfh_pct_white,
    drop column if exists nfh_pct_black,
    drop column if exists nfh_pct_asian,
    drop column if exists nfh_pct_hispanic,
    drop column if exists nfh_goal1_fin_services_score,
    drop column if exists nfh_goal2_goods_services_score,
    drop column if exists nfh_goal3_jobs_income_score,
    drop column if exists nfh_goal4_fin_shocks_score,
    drop column if exists nfh_goal5_build_assets_score,
    drop column if exists nfh_overall_score;


-- ============================================================
-- 0007_match_neighborhoods_rpc.sql
-- ============================================================

-- Replace match_neighborhoods to match the post-0006 schema:
--   * NFH columns are gone, so drop min_nfh_overall_score / min_nfh_goal4_score.
--   * competitive_score and shooting_incident_count are now first-class, so
--     accept max bounds for them and return competitive_score in the row.

drop function if exists public.match_neighborhoods(
    vector, text[], integer, double precision, double precision, integer,
    double precision, double precision, double precision, integer
);

create or replace function public.match_neighborhoods(
    query_embedding             vector(1024),
    boroughs                    text[]            default null,
    min_subway_station_count    integer           default null,
    min_avg_pedestrian          double precision  default null,
    min_storefront_density      double precision  default null,
    min_storefront_filing_count integer           default null,
    min_commercial_activity     double precision  default null,
    max_competitive_score       double precision  default null,
    max_shooting_incident_count integer           default null,
    match_count                 integer           default 50
)
returns table (
    neighborhood                text,
    cd                          text,
    borough                     text,
    commercial_activity_score   double precision,
    transit_activity_score      double precision,
    competitive_score           double precision,
    shooting_incident_count     integer,
    avg_pedestrian              double precision,
    subway_station_count        integer,
    storefront_filing_count     integer,
    storefront_density_per_km2  double precision,
    similarity                  double precision
)
language sql
stable
as $$
    select
        n.neighborhood,
        n.cd,
        n.borough,
        n.commercial_activity_score,
        n.transit_activity_score,
        n.competitive_score,
        n.shooting_incident_count,
        n.avg_pedestrian,
        n.subway_station_count,
        n.storefront_filing_count,
        n.storefront_density_per_km2,
        1 - (n.embedding <=> query_embedding) as similarity
    from public.neighborhoods n
    where n.embedding is not null
      and (boroughs is null or n.borough = any (boroughs))
      and (min_subway_station_count is null or n.subway_station_count >= min_subway_station_count)
      and (min_avg_pedestrian is null or n.avg_pedestrian >= min_avg_pedestrian)
      and (min_storefront_density is null or n.storefront_density_per_km2 >= min_storefront_density)
      and (min_storefront_filing_count is null or n.storefront_filing_count >= min_storefront_filing_count)
      and (min_commercial_activity is null or n.commercial_activity_score >= min_commercial_activity)
      and (max_competitive_score is null or n.competitive_score <= max_competitive_score)
      and (max_shooting_incident_count is null or n.shooting_incident_count <= max_shooting_incident_count)
    order by n.embedding <=> query_embedding
    limit match_count;
$$;

grant execute on function public.match_neighborhoods(
    vector, text[], integer, double precision, double precision, integer,
    double precision, double precision, integer, integer
) to anon, authenticated;


-- ============================================================
-- 0008_nfh_columns.sql
-- ============================================================

-- Re-introduce NFH (Neighborhood Financial Health) columns dropped by 0006.
-- The pipeline now produces them again now that the NFH raw CSV is in data/raw/,
-- and the Ranking page exposes Min NFH Goal 4 / Min NFH Overall sliders.

alter table public.neighborhoods
    add column if not exists nfh_median_income           double precision,
    add column if not exists nfh_poverty_rate            double precision,
    add column if not exists nfh_pct_white               double precision,
    add column if not exists nfh_pct_black               double precision,
    add column if not exists nfh_pct_asian               double precision,
    add column if not exists nfh_pct_hispanic            double precision,
    add column if not exists nfh_goal1_fin_services_score double precision,
    add column if not exists nfh_goal2_goods_services_score double precision,
    add column if not exists nfh_goal3_jobs_income_score double precision,
    add column if not exists nfh_goal4_fin_shocks_score  double precision,
    add column if not exists nfh_goal5_build_assets_score double precision,
    add column if not exists nfh_overall_score           double precision;


-- ============================================================
-- 0009_match_neighborhoods_rpc.sql
-- ============================================================

-- Add Min NFH Goal 4 / Min NFH Overall threshold args back to match_neighborhoods
-- and return the two scores so the API can surface them.

drop function if exists public.match_neighborhoods(
    vector, text[], integer, double precision, double precision, integer,
    double precision, double precision, integer, integer
);

create or replace function public.match_neighborhoods(
    query_embedding             vector(1024),
    boroughs                    text[]            default null,
    min_subway_station_count    integer           default null,
    min_avg_pedestrian          double precision  default null,
    min_storefront_density      double precision  default null,
    min_storefront_filing_count integer           default null,
    min_commercial_activity     double precision  default null,
    max_competitive_score       double precision  default null,
    max_shooting_incident_count integer           default null,
    min_nfh_goal4_score         double precision  default null,
    min_nfh_overall_score       double precision  default null,
    match_count                 integer           default 50
)
returns table (
    neighborhood                text,
    cd                          text,
    borough                     text,
    commercial_activity_score   double precision,
    transit_activity_score      double precision,
    competitive_score           double precision,
    shooting_incident_count     integer,
    avg_pedestrian              double precision,
    subway_station_count        integer,
    storefront_filing_count     integer,
    storefront_density_per_km2  double precision,
    nfh_goal4_fin_shocks_score  double precision,
    nfh_overall_score           double precision,
    similarity                  double precision
)
language sql
stable
as $$
    select
        n.neighborhood,
        n.cd,
        n.borough,
        n.commercial_activity_score,
        n.transit_activity_score,
        n.competitive_score,
        n.shooting_incident_count,
        n.avg_pedestrian,
        n.subway_station_count,
        n.storefront_filing_count,
        n.storefront_density_per_km2,
        n.nfh_goal4_fin_shocks_score,
        n.nfh_overall_score,
        1 - (n.embedding <=> query_embedding) as similarity
    from public.neighborhoods n
    where n.embedding is not null
      and (boroughs is null or n.borough = any (boroughs))
      and (min_subway_station_count is null or n.subway_station_count >= min_subway_station_count)
      and (min_avg_pedestrian is null or n.avg_pedestrian >= min_avg_pedestrian)
      and (min_storefront_density is null or n.storefront_density_per_km2 >= min_storefront_density)
      and (min_storefront_filing_count is null or n.storefront_filing_count >= min_storefront_filing_count)
      and (min_commercial_activity is null or n.commercial_activity_score >= min_commercial_activity)
      and (max_competitive_score is null or n.competitive_score <= max_competitive_score)
      and (max_shooting_incident_count is null or n.shooting_incident_count <= max_shooting_incident_count)
      and (min_nfh_goal4_score is null or n.nfh_goal4_fin_shocks_score >= min_nfh_goal4_score)
      and (min_nfh_overall_score is null or n.nfh_overall_score >= min_nfh_overall_score)
    order by n.embedding <=> query_embedding
    limit match_count;
$$;

grant execute on function public.match_neighborhoods(
    vector, text[], integer, double precision, double precision, integer,
    double precision, double precision, integer, double precision, double precision, integer
) to anon, authenticated;


-- Force PostgREST to reload its schema cache so the API sees the new table immediately.
notify pgrst, 'reload schema';
