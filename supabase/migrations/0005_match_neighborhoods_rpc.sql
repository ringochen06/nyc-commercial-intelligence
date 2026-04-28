-- RPC for the Ranking page: take a query embedding plus an optional set of hard
-- filters, and return rows ordered by cosine similarity. Doing the cosine math in
-- Postgres avoids shipping all 1536-dim vectors over the wire to the Vercel function.

create or replace function public.match_neighborhoods(
    query_embedding             vector(1536),
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
