-- Master neighborhood feature table.
-- One row per CDTA (Community District Tabulation Area), mirroring
-- data/processed/neighborhood_features_final.csv plus an OpenAI
-- text-embedding-3-small (1536-dim) vector built from the row's text profile.

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

    embedding                                   vector(1536),
    embedding_text                              text,

    pipeline_loaded_at                          timestamptz not null default now()
);

-- The CSV has two columns that differ only in case: act_OTHER_storefront and
-- act_other_storefront. Postgres folds identifiers to lowercase, so the loader
-- maps the lowercase ("act_other") variant to *_lower_storefront / *_lower_density.

comment on table public.neighborhoods is
    'One row per NYC CDTA. Source: data/processed/neighborhood_features_final.csv plus OpenAI embeddings of the row text profile.';
comment on column public.neighborhoods.cd is
    'CDTA 2020 code (e.g. MN01). Joins to nycdta2020 GeoJSON used for the choropleth.';
comment on column public.neighborhoods.embedding is
    'OpenAI text-embedding-3-small vector of embedding_text. Cosine distance via the <=> operator.';
comment on column public.neighborhoods.embedding_text is
    'Exact text profile used to produce embedding. Stored so we can rebuild vectors deterministically.';
