-- pgvector for embedding similarity search.
-- Supabase already exposes the extension; this just enables it in the public schema.
create extension if not exists vector;
