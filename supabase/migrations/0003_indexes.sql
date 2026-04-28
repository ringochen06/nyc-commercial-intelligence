-- Borough is the most common hard filter; index it even though the table is small (~70 rows).
create index neighborhoods_borough_idx
    on public.neighborhoods (borough);

-- HNSW index for cosine similarity. At ~70 rows a sequential scan is actually fastest,
-- so this is mainly forward-looking; pgvector still uses it correctly when it helps.
-- Switch to ivfflat with `lists = sqrt(rowcount)` if the table grows past a few thousand rows.
create index neighborhoods_embedding_hnsw_idx
    on public.neighborhoods
    using hnsw (embedding vector_cosine_ops);
