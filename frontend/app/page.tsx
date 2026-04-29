"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api } from "@/lib/api";
import type { FeatureRangesResponse } from "@/lib/types";
import { SectionCard } from "@/components/SectionCard";

const TOTAL_CDTAS = 71;

interface Stat {
  label: string;
  value: string;
  hint?: string;
}

function StatTile({ stat }: { stat: Stat }) {
  return (
    <div className="glass-card-tight p-5">
      <div className="text-[11px] uppercase tracking-[0.16em] text-muted font-medium">
        {stat.label}
      </div>
      <div className="text-3xl font-semibold text-ink tracking-tightish mt-2 tabular-nums">
        {stat.value}
      </div>
      {stat.hint && (
        <div className="text-[12px] text-muted mt-1.5">{stat.hint}</div>
      )}
    </div>
  );
}

interface ActionCardProps {
  href: string;
  step: string;
  title: string;
  description: string;
  bullets: string[];
}

function ActionCard({ href, step, title, description, bullets }: ActionCardProps) {
  return (
    <Link href={href} className="glass-card card-link p-6 flex flex-col h-full">
      <p className="text-[12px] uppercase tracking-[0.18em] text-muted font-medium">
        {step}
      </p>
      <h3 className="text-xl font-semibold text-ink mt-2 tracking-tightish">
        {title}
      </h3>
      <p className="text-[14px] leading-6 text-muted mt-2">{description}</p>
      <ul className="mt-4 space-y-1.5 text-[13px] text-ink/80">
        {bullets.map((b) => (
          <li key={b} className="flex items-start gap-2">
            <span
              aria-hidden
              className="mt-2 h-1.5 w-1.5 rounded-full shrink-0"
              style={{
                background: "linear-gradient(135deg, #fb923c, #f59e0b)",
              }}
            />
            <span className="leading-5">{b}</span>
          </li>
        ))}
      </ul>
      <div className="mt-auto pt-5 flex items-center gap-1.5 text-[13px] font-medium text-ink">
        Open
        <span
          aria-hidden
          className="card-link-arrow inline-block"
          style={{ transition: "transform 0.32s var(--ease-spring)" }}
        >
          →
        </span>
      </div>
    </Link>
  );
}

export default function HomePage() {
  const [ranges, setRanges] = useState<FeatureRangesResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .featureRanges()
      .then(setRanges)
      .catch((e) => setError(e.message));
  }, []);

  const numericFeatureCount = ranges
    ? Object.keys(ranges.ranges).length
    : null;

  const stats: Stat[] = [
    {
      label: "CDTAs",
      value: TOTAL_CDTAS.toString(),
      hint: "Community District Tabulation Areas",
    },
    {
      label: "Boroughs",
      value: ranges ? ranges.boroughs.length.toString() : "—",
      hint: ranges ? ranges.boroughs.map((b) => b.charAt(0) + b.slice(1).toLowerCase()).join(" · ") : "Loading",
    },
    {
      label: "Activity categories",
      value: ranges ? ranges.activity_columns.length.toString() : "—",
      hint: "act_*_storefront columns",
    },
    {
      label: "Numeric features",
      value: numericFeatureCount !== null ? numericFeatureCount.toString() : "—",
      hint: "Filterable in /api/feature-ranges",
    },
  ];

  return (
    <div className="space-y-10">
      {/* Hero */}
      <header className="space-y-3 max-w-3xl">
        <p className="text-[12px] uppercase tracking-[0.18em] text-muted font-medium">
          Dashboard
        </p>
        <h1 className="text-4xl md:text-5xl font-semibold text-ink tracking-tightish leading-[1.05]">
          NYC Commercial
          <br />
          Intelligence
        </h1>
        <p className="text-[15px] leading-7 text-muted">
          Decision support for commercial site selection across all{" "}
          {TOTAL_CDTAS} New York City CDTAs. Combine deterministic filters with
          natural-language soft preferences to surface the right neighborhoods.
        </p>
      </header>

      {/* Stat strip */}
      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map((s) => (
          <StatTile key={s.label} stat={s} />
        ))}
      </section>

      {error && (
        <div className="glass-card-tight p-4 text-[13px] text-amber-900">
          Couldn&rsquo;t reach the API ({error}). Stats may be incomplete.
        </div>
      )}

      {/* Two big action cards */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ActionCard
          href="/k-selection"
          step="Step 1"
          title="K-Selection & Clustering"
          description="Group neighborhoods by the metrics you care about. Choose your cluster count — it carries over to Ranking."
          bullets={[
            "Custom NumPy K-Means with elbow + silhouette",
            "Pick which features to cluster on",
            "Borough-aware filtering and a CDTA choropleth",
          ]}
        />
        <ActionCard
          href="/ranking"
          step="Step 2"
          title="Ranking"
          description="Apply hard SQL filters, then blend semantic similarity with a competition penalty to rank neighborhoods for your site."
          bullets={[
            "Hard filters via DuckDB SQL (transparent query)",
            "Soft preferences via OpenAI / Supabase pgvector",
            "Optional Claude analysis on the filtered set",
          ]}
        />
      </section>

      {/* How it works */}
      <SectionCard
        title="How the pipeline fits together"
        caption="Public datasets → engineered features → semantic profiles → blended ranking."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <PipelineStep
            num="01"
            title="Ingest"
            text="MTA subway, DOT pedestrian counts, NYC storefront filings, NYS shooting incidents, ACS demographics."
          />
          <PipelineStep
            num="02"
            title="Aggregate"
            text="Spatial join into 2020 CDTA polygons. Borough/citywide median imputation for missing rows."
          />
          <PipelineStep
            num="03"
            title="Embed"
            text="Per-CDTA text profiles → 1536-d OpenAI embeddings, stored in Supabase with HNSW cosine index."
          />
          <PipelineStep
            num="04"
            title="Rank"
            text="MinMax([semantic, −competition]) on the filtered set, then α·semantic + (1−α)·competition."
          />
        </div>
      </SectionCard>
    </div>
  );
}

function PipelineStep({
  num,
  title,
  text,
}: {
  num: string;
  title: string;
  text: string;
}) {
  return (
    <div className="glass-card-tight p-4">
      <div className="text-[11px] tabular-nums text-muted font-medium">{num}</div>
      <div className="text-[14px] font-semibold text-ink mt-1">{title}</div>
      <div className="text-[13px] leading-5 text-muted mt-1.5">{text}</div>
    </div>
  );
}
