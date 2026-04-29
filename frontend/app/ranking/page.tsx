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

const DEFAULT_QUERY =
  "quiet residential area suitable for boutique retail with good subway access";

const fmt = (v: number | null | undefined, digits: number): string =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

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
      <div className="text-sm text-muted">
        {error ? `Error: ${error}` : "Loading…"}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-ink">Ranking</h1>
        <p className="text-sm text-muted mt-1">
          Hard filters (DuckDB SQL), then α·semantic + β·competitive penalty
          (MinMax on the filtered set). Cluster columns appear when you have run
          K-Selection Analysis on the home page (saved to your browser).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
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
            <div className="space-y-3">
              <textarea
                className="w-full border rounded p-2 text-sm"
                rows={4}
                value={draftQuery}
                onChange={(e) => setDraftQuery(e.target.value)}
              />
              <button
                onClick={() => setCommittedQuery(draftQuery)}
                className="bg-ink text-white px-3 py-1.5 rounded text-sm"
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
                <div className="text-sm font-medium mb-1">
                  Competitive score source
                </div>
                <select
                  className="w-full border rounded px-2 py-1 text-sm"
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
              <p className="text-xs text-muted">
                Using clusters from your last K-Selection run (k = {cluster.k},
                {Object.keys(cluster.assignments).length} neighborhoods).
                <button
                  className="ml-2 underline"
                  onClick={() => cluster.clear()}
                >
                  clear
                </button>
              </p>
            ) : (
              <p className="text-xs text-muted">
                No clustering on file. Run{" "}
                <a className="underline" href="/">
                  K-Selection
                </a>{" "}
                to fill cluster columns.
              </p>
            )}
          </SectionCard>
        </aside>

        <div className="space-y-6">
          <SectionCard
            title={
              result
                ? `Ranked neighborhoods (${result.n_filtered} of ${result.n_total})`
                : "Ranked neighborhoods"
            }
            caption={
              loading
                ? "Computing…"
                : error
                  ? `Error: ${error}`
                  : "MinMax([semantic, -competitive score]) on the filtered set, then α·semantic + (1−α)·competitive penalty."
            }
          >
            {result && result.rows.length > 0 ? (
              <div className="overflow-auto max-h-[480px]">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Neighborhood</th>
                      <th>Borough</th>
                      {cluster.k !== null && <th>Cluster</th>}
                      {cluster.k !== null && <th>Cluster description</th>}
                      <th>Semantic</th>
                      <th>Specific competitive</th>
                      <th>Blended</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.rows.map((row) => (
                      <tr key={row.neighborhood}>
                        <td>{row.rank}</td>
                        <td>{row.neighborhood}</td>
                        <td>{row.borough || ""}</td>
                        {cluster.k !== null && (
                          <td>{row.cluster ?? ""}</td>
                        )}
                        {cluster.k !== null && (
                          <td className="text-xs text-muted">
                            {row.cluster_description || ""}
                          </td>
                        )}
                        <td>{fmt(row.semantic_similarity, 4)}</td>
                        <td>{fmt(row.specific_competitive_score, 3)}</td>
                        <td className="font-medium">
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
            title="AI analysis (Claude)"
            caption="Optional. Server must have ANTHROPIC_API_KEY set."
          >
            <button
              onClick={askClaude}
              disabled={agentLoading}
              className="bg-ink text-white px-3 py-1.5 rounded text-sm disabled:opacity-50"
            >
              {agentLoading ? "Asking Claude…" : "Ask Claude to analyze"}
            </button>
            {agentError && (
              <p className="text-sm text-red-600 mt-3">{agentError}</p>
            )}
            {agentAnswer && (
              <pre className="mt-3 whitespace-pre-wrap text-sm bg-slate-50 p-3 rounded border border-slate-200">
                {agentAnswer}
              </pre>
            )}
          </SectionCard>

          <SectionCard title="Generated SQL" caption="DuckDB query against `neighborhoods`.">
            <pre className="text-xs bg-slate-50 p-3 rounded border border-slate-200 overflow-auto">
              {result?.sql || "—"}
            </pre>
          </SectionCard>
        </div>
      </div>
    </div>
  );
}
