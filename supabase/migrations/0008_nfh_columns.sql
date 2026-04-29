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
