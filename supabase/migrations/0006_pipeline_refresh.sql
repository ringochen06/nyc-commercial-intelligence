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
