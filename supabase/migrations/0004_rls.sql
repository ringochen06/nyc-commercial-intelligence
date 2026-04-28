-- Row-level security: the public anon key is allowed to SELECT the neighborhoods table
-- (the dashboard is a read-only public app). All writes go through the service-role key
-- used by the loader script and are therefore exempt from RLS.

alter table public.neighborhoods enable row level security;

create policy "neighborhoods_anon_read"
    on public.neighborhoods
    for select
    to anon, authenticated
    using (true);
