"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useClusterStore } from "@/lib/state";
import type {
  CdtaGeoResponse,
  FeatureRangesResponse,
  HardFilters,
  RankResponse,
} from "@/lib/types";
import { MultiSelect } from "@/components/MultiSelect";
import { PlotlyChart } from "@/components/PlotlyChart";
import { SectionCard } from "@/components/SectionCard";
import { Slider } from "@/components/Slider";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const DEFAULT_QUERY =
  "quiet residential area suitable for boutique retail with good subway access";

const fmt = (v: number | null | undefined, digits: number): string =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

const MARKDOWN_COMPONENTS: Components = {
  h1: (props) => <h1 className="text-base font-semibold mt-4 mb-2 first:mt-0" {...props} />,
  h2: (props) => <h2 className="text-sm font-semibold mt-4 mb-2 first:mt-0" {...props} />,
  h3: (props) => <h3 className="text-sm font-semibold mt-3 mb-1.5 first:mt-0" {...props} />,
  p: (props) => <p className="text-sm leading-6 my-2 first:mt-0 last:mb-0" {...props} />,
  ul: (props) => <ul className="list-disc pl-5 my-2 space-y-1 text-sm" {...props} />,
  ol: (props) => <ol className="list-decimal pl-5 my-2 space-y-1 text-sm" {...props} />,
  li: (props) => <li className="leading-6" {...props} />,
  strong: (props) => <strong className="font-semibold" {...props} />,
  em: (props) => <em className="italic" {...props} />,
  a: (props) => (
    <a
      className="text-blue-700 underline underline-offset-2"
      target="_blank"
      rel="noreferrer"
      {...props}
    />
  ),
  blockquote: (props) => (
    <blockquote
      className="border-l-2 border-slate-300 pl-3 my-2 text-slate-700"
      {...props}
    />
  ),
  code: ({ className, children, ...props }) => {
    const isBlock = /language-/.test(className ?? "");
    if (isBlock) {
      return (
        <code className={`${className ?? ""} text-xs`} {...props}>
          {children}
        </code>
      );
    }
    return (
      <code
        className="bg-slate-200 text-slate-900 rounded px-1 py-0.5 text-[0.8rem] font-mono"
        {...props}
      >
        {children}
      </code>
    );
  },
  pre: (props) => (
    <pre
      className="bg-slate-900 text-slate-100 rounded p-3 my-3 overflow-x-auto text-xs"
      {...props}
    />
  ),
  table: (props) => (
    <div className="overflow-x-auto my-3">
      <table className="text-xs border-collapse" {...props} />
    </div>
  ),
  th: (props) => (
    <th
      className="border border-slate-300 px-2 py-1 bg-slate-100 text-left font-semibold"
      {...props}
    />
  ),
  td: (props) => <td className="border border-slate-300 px-2 py-1 align-top" {...props} />,
  hr: () => <hr className="my-3 border-slate-200" />,
};

export default function RankingPage() {
  const [ranges, setRanges] = useState<FeatureRangesResponse | null>(null);
  const [geo, setGeo] = useState<CdtaGeoResponse | null>(null);

  // Hard filters
  const [boroughs, setBoroughs] = useState<string[]>([]);
  const [minSubway, setMinSubway] = useState(0);
  const [minPed, setMinPed] = useState(0);
  const [minDensity, setMinDensity] = useState(0);
  const [minFilings, setMinFilings] = useState(0);
  const [minCommercial, setMinCommercial] = useState(0);
  const [maxCompetitive, setMaxCompetitive] = useState(0);
  const [maxShootingIncident, setMaxShootingIncident] = useState(0);
  const [minNfhGoal4, setMinNfhGoal4] = useState<number | null>(null);
  const [minNfhOverall, setMinNfhOverall] = useState<number | null>(null);

  // Soft preferences
  const [draftQuery, setDraftQuery] = useState(DEFAULT_QUERY);
  const [committedQuery, setCommittedQuery] = useState(DEFAULT_QUERY);
  const [alpha, setAlpha] = useState(0.8);
  const [competitiveSource, setCompetitiveSource] = useState<string>("__overall__");

  const [result, setResult] = useState<RankResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Optional Claude
  const [agentLoading, setAgentLoading] = useState(false);
  const [agentAnswer, setAgentAnswer] = useState<string | null>(null);
  const [agentError, setAgentError] = useState<string | null>(null);

  const cluster = useClusterStore();

  useEffect(() => {
    api
      .featureRanges()
      .then((r) => {
        setRanges(r);
        setBoroughs(r.boroughs);
        const init = (key: string, fallback = 0): number => {
          const v = r.ranges[key];
          return v ? v.min : fallback;
        };
        setMinSubway(init("subway_station_count"));
        setMinPed(init("avg_pedestrian"));
        setMinDensity(init("storefront_density_per_km2"));
        setMinFilings(init("storefront_filing_count"));
        setMinCommercial(init("commercial_activity_score"));
        setMaxCompetitive(
          r.ranges["competitive_score"] ? r.ranges["competitive_score"].max : 0,
        );
        setMaxShootingIncident(
          r.ranges["shooting_incident_count"]
            ? r.ranges["shooting_incident_count"].max
            : 0,
        );
        setMinNfhGoal4(
          r.has_nfh_goal4 && r.ranges["nfh_goal4_fin_shocks_score"]
            ? r.ranges["nfh_goal4_fin_shocks_score"].min
            : null,
        );
        setMinNfhOverall(
          r.has_nfh_overall && r.ranges["nfh_overall_score"]
            ? r.ranges["nfh_overall_score"].min
            : null,
        );
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    api.cdtaGeo().then(setGeo).catch(() => {});
  }, []);

  const filters: HardFilters = useMemo(
    () => ({
      boroughs,
      min_subway_stations: minSubway,
      min_avg_pedestrian: minPed,
      min_storefront_density: minDensity,
      min_storefront_filings: minFilings,
      min_commercial_activity: minCommercial,
      max_competitive_score: maxCompetitive,
      max_shooting_incident_count: maxShootingIncident,
      min_nfh_goal4: minNfhGoal4 ?? undefined,
      min_nfh_overall: minNfhOverall ?? undefined,
    }),
    [
      boroughs,
      minSubway,
      minPed,
      minDensity,
      minFilings,
      minCommercial,
      maxCompetitive,
      maxShootingIncident,
      minNfhGoal4,
      minNfhOverall,
    ],
  );

  // Auto re-rank when filters / alpha / committed query change
  useEffect(() => {
    if (!ranges) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .rank({
        query: committedQuery,
        alpha,
        filters,
        vintage: "present",
        competitive_source: competitiveSource,
        cluster_assignments: cluster.assignments,
        cluster_briefs: cluster.briefs,
      })
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [
    ranges,
    filters,
    alpha,
    committedQuery,
    competitiveSource,
    cluster.assignments,
    cluster.briefs,
  ]);

  const r = ranges?.ranges;

  const askClaude = async () => {
    setAgentLoading(true);
    setAgentError(null);
    setAgentAnswer(null);
    try {
      const r = await api.agent({
        query: committedQuery,
        alpha,
        filters,
        vintage: "present",
        competitive_source: competitiveSource,
        cluster_assignments: cluster.assignments,
        cluster_briefs: cluster.briefs,
      });
      setAgentAnswer(r.answer);
    } catch (e: any) {
      setAgentError(e.message);
    } finally {
      setAgentLoading(false);
    }
  };

  const mapPlot = useMemo(() => {
    if (!result || !geo || !geo.geojson.features.length || !result.rows.length)
      return null;
    const rows = result.rows.filter(
      (row) => row.map_key && Number.isFinite(row.blended_score),
    );
    if (!rows.length) return null;
    const zmin = Math.min(...rows.map((r) => r.blended_score));
    const zmax = Math.max(...rows.map((r) => r.blended_score));
    const trace: any = {
      type: "choroplethmapbox",
      geojson: geo.geojson,
      locations: rows.map((r) => r.map_key as string),
      z: rows.map((r) => r.blended_score),
      featureidkey: "properties.map_key",
      colorscale: [
        [0, "#e8f5e9"],
        [0.35, "#a5d6a7"],
        [0.65, "#43a047"],
        [1, "#1b5e20"],
      ],
      zmin,
      zmax: Math.max(zmax, zmin + 1e-9),
      marker: { opacity: 0.82, line: { width: 0.6, color: "white" } },
      colorbar: { title: { text: "Blended" }, tickformat: ".3f" },
      text: rows.map(
        (r) =>
          `${r.neighborhood}<br>rank ${r.rank}<br>blended ${fmt(r.blended_score, 3)}`,
      ),
      hovertemplate: "<b>%{text}</b><extra></extra>",
    };
    return (
      <PlotlyChart
        data={[trace]}
        layout={{
          height: 480,
          margin: { l: 0, r: 0, t: 8, b: 0 },
          mapbox: {
            style: "open-street-map",
            center: { lat: geo.center.lat, lon: geo.center.lon },
            zoom: 9,
          } as any,
        }}
      />
    );
  }, [result, geo]);

  if (!ranges) {
    return (
      <div className="glass-card p-6 text-sm text-muted">
        {error ? `Error: ${error}` : "Loading…"}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <p className="text-[12px] uppercase tracking-[0.18em] text-muted font-medium">
          Step 2
        </p>
        <h1 className="text-3xl md:text-4xl font-semibold text-ink tracking-tightish">
          Ranking
        </h1>
        <p className="text-[14px] leading-6 text-muted max-w-2xl">
          Apply deterministic SQL filters, then blend semantic similarity with a
          competition penalty. Cluster columns appear once you&rsquo;ve run
          K-Selection on the home page.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[340px_1fr] gap-6">
        <aside className="space-y-5">
          <SectionCard title="Hard Filters">
            <div className="space-y-4">
              <MultiSelect
                label="Borough"
                options={ranges.boroughs}
                value={boroughs}
                onChange={setBoroughs}
              />

              {r?.["subway_station_count"] && (
                <Slider
                  label="Min subway stations"
                  value={minSubway}
                  min={Math.floor(r["subway_station_count"].min)}
                  max={Math.ceil(r["subway_station_count"].max)}
                  onChange={setMinSubway}
                />
              )}
              {r?.["avg_pedestrian"] && (
                <Slider
                  label="Min avg pedestrians"
                  value={minPed}
                  min={Math.floor(r["avg_pedestrian"].min)}
                  max={Math.ceil(r["avg_pedestrian"].max)}
                  onChange={setMinPed}
                />
              )}
              {r?.["storefront_density_per_km2"] && (
                <Slider
                  label="Min storefront density (per km²)"
                  value={minDensity}
                  min={r["storefront_density_per_km2"].min}
                  max={r["storefront_density_per_km2"].max}
                  step={0.5}
                  onChange={setMinDensity}
                  format={(v) => v.toFixed(1)}
                />
              )}
              {r?.["storefront_filing_count"] && (
                <Slider
                  label="Min storefront filings"
                  value={minFilings}
                  min={Math.floor(r["storefront_filing_count"].min)}
                  max={Math.ceil(r["storefront_filing_count"].max)}
                  onChange={setMinFilings}
                />
              )}
              {r?.["commercial_activity_score"] && (
                <Slider
                  label="Min commercial activity score"
                  value={minCommercial}
                  min={r["commercial_activity_score"].min}
                  max={r["commercial_activity_score"].max}
                  step={Math.max(
                    0.001,
                    Math.min(
                      0.5,
                      (r["commercial_activity_score"].max -
                        r["commercial_activity_score"].min) /
                        200,
                    ),
                  )}
                  onChange={setMinCommercial}
                  format={(v) => v.toFixed(3)}
                />
              )}
              {r?.["competitive_score"] && (
                <Slider
                  label="Max competitive score"
                  value={maxCompetitive}
                  min={r["competitive_score"].min}
                  max={r["competitive_score"].max}
                  step={
                    Math.max(
                      0.001,
                      Math.min(
                        0.5,
                        (r["competitive_score"].max - r["competitive_score"].min) /
                          200,
                      ),
                    )
                  }
                  onChange={setMaxCompetitive}
                  format={(v) => v.toFixed(3)}
                />
              )}
              {r?.["shooting_incident_count"] && (
                <Slider
                  label="Max shooting incident count"
                  value={maxShootingIncident}
                  min={r["shooting_incident_count"].min}
                  max={r["shooting_incident_count"].max}
                  step={1}
                  onChange={setMaxShootingIncident}
                  format={(v) => v.toFixed(0)}
                />
              )}
              {ranges.has_nfh_goal4 &&
                r?.["nfh_goal4_fin_shocks_score"] &&
                minNfhGoal4 !== null && (
                  <Slider
                    label="Min NFH Goal 4 (shocks)"
                    value={minNfhGoal4}
                    min={r["nfh_goal4_fin_shocks_score"].min}
                    max={r["nfh_goal4_fin_shocks_score"].max}
                    step={0.1}
                    onChange={setMinNfhGoal4}
                    format={(v) => v.toFixed(2)}
                  />
                )}
              {ranges.has_nfh_overall &&
                r?.["nfh_overall_score"] &&
                minNfhOverall !== null && (
                  <Slider
                    label="Min NFH overall"
                    value={minNfhOverall}
                    min={r["nfh_overall_score"].min}
                    max={r["nfh_overall_score"].max}
                    step={0.1}
                    onChange={setMinNfhOverall}
                    format={(v) => v.toFixed(2)}
                  />
                )}
            </div>
          </SectionCard>

          <SectionCard title="Soft Preferences">
            <div className="space-y-4">
              <textarea
                className="glass-input"
                rows={4}
                value={draftQuery}
                onChange={(e) => setDraftQuery(e.target.value)}
                placeholder="Describe your ideal neighborhood…"
              />
              <button
                onClick={() => setCommittedQuery(draftQuery)}
                className="btn-primary"
              >
                Update soft ranking
              </button>
              <Slider
                label="α (semantic) — β = 1 − α"
                value={alpha}
                min={0}
                max={1}
                step={0.05}
                onChange={setAlpha}
                format={(v) => `α=${v.toFixed(2)}, β=${(1 - v).toFixed(2)}`}
              />
              <div>
                <div className="text-[13px] font-medium text-ink mb-2">
                  Competitive score source
                </div>
                <select
                  className="glass-select"
                  value={competitiveSource}
                  onChange={(e) => setCompetitiveSource(e.target.value)}
                >
                  <option value="__overall__">
                    Overall (all storefront filings)
                  </option>
                  {ranges.activity_columns.map((col) => (
                    <option key={col} value={col}>
                      {col
                        .replace(/^act_/, "")
                        .replace(/_storefront$/, "")
                        .replaceAll("_", " ")}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </SectionCard>

          <SectionCard title="Cluster context">
            {cluster.k ? (
              <p className="text-[13px] text-muted leading-5">
                Using clusters from your last K-Selection run (k = {cluster.k},{" "}
                {Object.keys(cluster.assignments).length} neighborhoods).
                <button
                  className="ml-2 underline underline-offset-2 hover:text-ink"
                  onClick={() => cluster.clear()}
                >
                  clear
                </button>
              </p>
            ) : (
              <p className="text-[13px] text-muted leading-5">
                No clustering on file. Run{" "}
                <a className="underline underline-offset-2 hover:text-ink" href="/k-selection">
                  K-Selection
                </a>{" "}
                to fill cluster columns.
              </p>
            )}
          </SectionCard>
        </aside>

        <div className="space-y-6">
          <SectionCard
            title="Ranked neighborhoods"
            caption={
              loading
                ? "Computing…"
                : error
                  ? `Error: ${error}`
                  : "MinMax([semantic, -competitive score]) on the filtered set, then α·semantic + (1−α)·competitive penalty."
            }
            actions={
              result && (
                <span className="text-[12px] text-muted tabular-nums">
                  {result.n_filtered} of {result.n_total}
                </span>
              )
            }
          >
            {result && result.rows.length > 0 ? (
              <div className="overflow-auto max-h-[520px] -mx-2 px-2">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Neighborhood</th>
                      <th>Borough</th>
                      {cluster.k !== null && <th>Cluster</th>}
                      {cluster.k !== null && <th>Cluster description</th>}
                      <th className="text-right">Semantic</th>
                      <th className="text-right">Competitive</th>
                      <th className="text-right">Blended</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.map((row) => (
                      <tr key={row.neighborhood}>
                        <td className="text-muted tabular-nums">{row.rank}</td>
                        <td className="font-medium text-ink">{row.neighborhood}</td>
                        <td className="text-muted">{row.borough || ""}</td>
                        {cluster.k !== null && (
                          <td className="tabular-nums">{row.cluster ?? ""}</td>
                        )}
                        {cluster.k !== null && (
                          <td className="text-[12px] text-muted">
                            {row.cluster_description || ""}
                          </td>
                        )}
                        <td className="text-right tabular-nums">
                          {fmt(row.semantic_similarity, 4)}
                        </td>
                        <td className="text-right tabular-nums">
                          {fmt(row.specific_competitive_score, 3)}
                        </td>
                        <td className="text-right tabular-nums font-semibold text-ink">
                          {fmt(row.blended_score, 4)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              !loading && (
                <p className="text-sm text-muted">
                  {error || "No rows to display."}
                </p>
              )
            )}
          </SectionCard>

          {mapPlot && (
            <SectionCard
              title="NYC map"
              caption="Blended score (light green = lower, dark green = higher on the filtered set)."
            >
              {mapPlot}
            </SectionCard>
          )}

          <SectionCard
            title="AI analysis"
            caption="Claude reads the filtered table and recommends top picks. Requires ANTHROPIC_API_KEY on the server."
          >
            <button
              onClick={askClaude}
              disabled={agentLoading}
              className="btn-primary"
            >
              {agentLoading ? "Asking Claude…" : "Ask Claude to analyse"}
            </button>
            {agentError && <p className="pill-error mt-3 inline-block">{agentError}</p>}
            {agentAnswer && (
              <div className="mt-4 glass-card-tight p-4">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={MARKDOWN_COMPONENTS}
                >
                  {agentAnswer}
                </ReactMarkdown>
              </div>
            )}
          </SectionCard>

          <SectionCard title="Generated SQL" caption="DuckDB query against `neighborhoods`.">
            <pre className="text-[12px] leading-5 bg-slate-900/95 text-slate-100 rounded-xl p-4 overflow-auto font-mono">
              {result?.sql || "—"}
            </pre>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
