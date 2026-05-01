"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { colorForCluster } from "@/lib/palette";
import { useClusterStore } from "@/lib/state";
import type {
  CdtaGeoResponse,
  ClusterResponse,
  FeatureRangesResponse,
} from "@/lib/types";
import { MultiSelect } from "@/components/MultiSelect";
import { PlotlyChart } from "@/components/PlotlyChart";
import { SectionCard } from "@/components/SectionCard";
import { Slider } from "@/components/Slider";

const BASE_CANDIDATE_FEATURES = [
  "storefront_filing_count",
  "avg_pedestrian",
  "subway_station_count",
  "storefront_density_per_km2",
  "commercial_activity_score",
  "competitive_score",
  "shooting_incident_count",
  "transit_activity_score",
  "category_entropy",
  "category_diversity",
  "peak_pedestrian",
  "subway_density_per_km2",
  "nfh_overall_score",
  "nfh_goal4_fin_shocks_score",
  "total_jobs",
];

const BASE_DEFAULT_FEATURES = [
  "storefront_filing_count",
  "avg_pedestrian",
  "subway_station_count",
  "storefront_density_per_km2",
  "commercial_activity_score",
  "competitive_score",
  "shooting_incident_count",
  "transit_activity_score",
  "category_entropy",
  "nfh_overall_score",
  "total_jobs",
];

const ACTIVITY_LABEL_REPLACEMENTS: Record<string, string> = {
  "accounting services": "accounting services",
  "broadcasting telecomm": "broadcasting and telecom",
  "educational services": "education",
  "finance and insurance": "finance and insurance",
  "food services": "food service",
  "health care or social assistance": "health care and social assistance",
  "information services": "information services",
  "legal services": "legal services",
  manufacturing: "manufacturing",
  "movies video sound": "media and entertainment",
  "no business activity identified": "no identified business activity",
  publishing: "publishing",
  "real estate": "real estate",
  retail: "retail",
  unknown: "unknown activity",
  wholesale: "wholesale",
  other: "other services",
};

const formatFeatureLabel = (feature: string): string => {
  if (feature.startsWith("act_") && feature.endsWith("_density")) {
    const base = feature
      .replace(/^act_/, "")
      .replace(/_density$/, "")
      .replace(/_/g, " ")
      .toLowerCase();
    return `${ACTIVITY_LABEL_REPLACEMENTS[base] ?? base} density`;
  }
  return feature.replace(/_/g, " ").toLowerCase();
};

const buildCandidateFeatures = (
  ranges: FeatureRangesResponse | null
): string[] => {
  if (!ranges) return BASE_CANDIDATE_FEATURES;
  const activityColumns = ranges.activity_columns ?? [];
  const activityDensity = activityColumns
    .map((col) => col.replace("_storefront", "_density"))
    .filter((col) => col.endsWith("_density"));
  return [...new Set([...BASE_CANDIDATE_FEATURES, ...activityDensity])];
};

const buildDefaultFeatures = (ranges: FeatureRangesResponse | null): string[] => {
  if (!ranges) return BASE_DEFAULT_FEATURES;
  const activityColumns = ranges.activity_columns ?? [];
  const activityDensity = activityColumns
    .map((col) => col.replace("_storefront", "_density"))
    .filter((col) => col.endsWith("_density"));
  return [...new Set([...BASE_DEFAULT_FEATURES, ...activityDensity])];
};

export default function KSelectionPage() {
  const [ranges, setRanges] = useState<FeatureRangesResponse | null>(null);
  const [boroughs, setBoroughs] = useState<string[]>([]);
  const [candidateFeatures, setCandidateFeatures] = useState<string[]>(
    BASE_CANDIDATE_FEATURES
  );
  const [features, setFeatures] = useState<string[]>(BASE_DEFAULT_FEATURES);
  const [maxK, setMaxK] = useState(8);
  /** Partition size for maps / Ranking (must lie in the swept k range). */
  const [chosenK, setChosenK] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClusterResponse | null>(null);
  const [geo, setGeo] = useState<CdtaGeoResponse | null>(null);
  const [scatterX, setScatterX] = useState<string>("avg_pedestrian");
  const [scatterY, setScatterY] = useState<string>("storefront_filing_count");

  const setClustering = useClusterStore((s) => s.setClustering);

  // Initial load: feature ranges + geo
  useEffect(() => {
    api
      .featureRanges()
      .then((r) => {
        setRanges(r);
        setBoroughs(r.boroughs);
        const candidate = buildCandidateFeatures(r);
        const defaults = buildDefaultFeatures(r);
        setCandidateFeatures(candidate);
        setFeatures(defaults);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    api.cdtaGeo().then(setGeo).catch(() => {});
  }, []);

  useEffect(() => {
    setChosenK((k) => Math.min(Math.max(k, 2), maxK));
  }, [maxK]);

  const fetchCluster = async (partitionK?: number) => {
    const kUse = partitionK ?? chosenK;
    setError(null);
    setLoading(true);
    try {
      const r = await api.cluster({
        features,
        boroughs,
        max_k: maxK,
        vintage: "present",
        chosen_k: kUse,
      });
      setResult(r);
      setChosenK(r.chosen_k);
      // Persist for the Ranking page.
      const assignments: Record<string, number> = {};
      for (const p of r.points) assignments[p.neighborhood] = p.cluster;
      const briefs: Record<string, string> = {};
      for (const s of r.cluster_summaries)
        briefs[String(s.cluster)] = s.description;
      setClustering(assignments, briefs, r.chosen_k, r.features);
      // Reset scatter axes if the new feature list shrank.
      if (!r.features.includes(scatterX)) setScatterX(r.features[0]);
      if (!r.features.includes(scatterY))
        setScatterY(r.features[Math.min(1, r.features.length - 1)]);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const elbowPlot = useMemo(() => {
    if (!result) return null;
    return (
      <PlotlyChart
        data={[
          {
            x: result.k_range,
            y: result.inertias,
            mode: "lines+markers",
            marker: { size: 8, color: "#4A90D9" },
            line: { width: 2, color: "#4A90D9" },
            name: "Inertia (WCSS)",
            yaxis: "y",
          },
          {
            x: result.k_range,
            y: result.silhouettes_numpy,
            mode: "lines+markers",
            marker: { size: 8, color: "#E74C3C", symbol: "square" },
            line: { width: 2, color: "#E74C3C", dash: "dash" },
            name: "Silhouette (NumPy)",
            yaxis: "y2",
          },
        ]}
        layout={{
          title: {
            text: `Elbow Method · ${result.points.length} neighborhoods × ${result.features.length} features`,
          },
          xaxis: { title: { text: "k" }, tickvals: result.k_range },
          yaxis: { title: { text: "Inertia (WCSS)" } },
          yaxis2: {
            title: { text: "Silhouette" },
            overlaying: "y",
            side: "right",
          },
          shapes: [
            {
              type: "line",
              x0: result.elbow_k,
              x1: result.elbow_k,
              y0: 0,
              y1: 1,
              yref: "paper",
              line: { dash: "dash", color: "gray" },
            },
            ...(result.elbow_k_kneedle !== result.elbow_k
              ? [
                  {
                    type: "line" as const,
                    x0: result.elbow_k_kneedle,
                    x1: result.elbow_k_kneedle,
                    y0: 0,
                    y1: 1,
                    yref: "paper" as const,
                    line: { dash: "dot" as const, color: "darkgreen" },
                  },
                ]
              : []),
          ],
          legend: { orientation: "h", y: 1.1, x: 1, xanchor: "right" },
          height: 460,
        }}
      />
    );
  }, [result]);

  const scatterPlot = useMemo(() => {
    if (!result) return null;
    const xi = result.features.indexOf(scatterX);
    const yi = result.features.indexOf(scatterY);
    if (xi < 0 || yi < 0) return null;

    const traces = [];
    for (let c = 0; c < result.chosen_k; c++) {
      const pts = result.points.filter((p) => p.cluster === c);
      traces.push({
        x: pts.map((p) => p.raw[scatterX]),
        y: pts.map((p) => p.raw[scatterY]),
        text: pts.map((p) => p.neighborhood),
        mode: "markers" as const,
        type: "scatter" as const,
        marker: {
          size: 9,
          color: colorForCluster(c),
          opacity: 0.85,
          line: { width: 0.5, color: "white" },
        },
        name: `Cluster ${c}`,
        hovertemplate:
          "<b>%{text}</b><br>" +
          formatFeatureLabel(scatterX) +
          ": %{x:.1f}<br>" +
          formatFeatureLabel(scatterY) +
          ": %{y:.1f}<extra></extra>",
      });

      // Centroid star (back to raw space)
      const cx =
        result.centroids_z[c][xi] * result.feature_stds[xi] +
        result.feature_means[xi];
      const cy =
        result.centroids_z[c][yi] * result.feature_stds[yi] +
        result.feature_means[yi];
      traces.push({
        x: [cx],
        y: [cy],
        mode: "markers" as const,
        type: "scatter" as const,
        marker: {
          size: 18,
          symbol: "star",
          color: colorForCluster(c),
          line: { width: 1.5, color: "black" },
        },
        showlegend: false,
        name: `Centroid ${c}`,
        hovertemplate: `<b>Centroid ${c}</b><br>%{x:.2f}, %{y:.2f}<extra></extra>`,
      });
    }
    return (
      <PlotlyChart
        data={traces}
        layout={{
          xaxis: { title: { text: formatFeatureLabel(scatterX) } },
          yaxis: { title: { text: formatFeatureLabel(scatterY) } },
          height: 420,
        }}
      />
    );
  }, [result, scatterX, scatterY]);

  const centroidBars = useMemo(() => {
    if (!result) return null;
    return (
      <PlotlyChart
        data={Array.from({ length: result.chosen_k }, (_, c) => ({
          type: "bar" as const,
          name: `Cluster ${c}`,
          x: result.features.map(formatFeatureLabel),
          y: result.centroids_z[c],
          marker: { color: colorForCluster(c), opacity: 0.85 },
        }))}
        layout={{
          barmode: "group",
          xaxis: { tickangle: -35 },
          yaxis: { title: { text: "z-score" } },
          height: 420,
        }}
      />
    );
  }, [result]);

  // For each cluster, the points closest to the centroid in z-space (most representative).
  const closestByCluster = useMemo(() => {
    const out: Record<number, { neighborhood: string; cd: string | null; distance: number }[]> = {};
    if (!result) return out;
    const k = result.chosen_k;
    const featureCount = result.features.length;
    for (let c = 0; c < k; c++) out[c] = [];
    for (const p of result.points) {
      const c = p.cluster;
      const centroid = result.centroids_z[c];
      let sumSq = 0;
      for (let i = 0; i < featureCount; i++) {
        const std = result.feature_stds[i] || 1;
        const z = (p.raw[result.features[i]] - result.feature_means[i]) / std;
        const diff = z - centroid[i];
        sumSq += diff * diff;
      }
      out[c].push({
        neighborhood: p.neighborhood,
        cd: p.cd,
        distance: Math.sqrt(sumSq),
      });
    }
    for (const c in out) out[c].sort((a, b) => a.distance - b.distance);
    return out;
  }, [result]);

  const mapPlot = useMemo(() => {
    if (!result || !geo || !geo.geojson.features.length) return null;
    const traces = [];
    for (let c = 0; c < result.chosen_k; c++) {
      const sub = result.points.filter((p) => p.cluster === c && p.map_key);
      if (!sub.length) continue;
      traces.push({
        type: "choroplethmapbox" as const,
        geojson: geo.geojson as any,
        locations: sub.map((p) => p.map_key as string),
        z: sub.map(() => 1),
        featureidkey: "properties.map_key",
        colorscale: [
          [0, colorForCluster(c)],
          [1, colorForCluster(c)],
        ] as any,
        showscale: false,
        marker: { opacity: 0.65, line: { width: 1, color: "white" } } as any,
        name: `Cluster ${c}`,
        text: sub.map((p) => `${p.neighborhood} (${p.cd})`),
        hovertemplate: `<b>%{text}</b><br>cluster=${c}<extra></extra>`,
      });
    }
    return (
      <PlotlyChart
        data={traces}
        layout={{
          height: 480,
          margin: { l: 0, r: 0, t: 8, b: 0 },
          mapbox: {
            style: "open-street-map",
            center: { lat: geo.center.lat, lon: geo.center.lon },
            zoom: 9,
          } as any,
          legend: { orientation: "h", y: 1.05, x: 1, xanchor: "right" },
        }}
      />
    );
  }, [result, geo]);

  return (
    <div className="space-y-8">
      <header className="space-y-2">
        <p className="text-[12px] uppercase tracking-[0.18em] text-muted font-medium">
          Step 1
        </p>
        <h1 className="text-3xl md:text-4xl font-semibold text-ink tracking-tightish">
          K-Selection &amp; Clustering
        </h1>
        <p className="text-[14px] leading-6 text-muted max-w-2xl">
          Group NYC&rsquo;s neighborhoods by the metrics that matter to you.
          Pick a cluster count — it carries over to the Ranking page automatically.
        </p>
      </header>

      <SectionCard title="Settings">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            {ranges && (
              <MultiSelect
                label="Borough"
                options={ranges.boroughs}
                value={boroughs}
                onChange={setBoroughs}
              />
            )}
          </div>

          <div className="space-y-4">
            <MultiSelect
              label="Features for clustering"
              options={candidateFeatures}
              value={features}
              onChange={setFeatures}
              format={formatFeatureLabel}
            />
            <Slider
              label="Maximum k"
              value={maxK}
              min={2}
              max={15}
              onChange={(v) => {
                setMaxK(v);
                setChosenK(v);
              }}
              hint="Upper bound for the k sweep. Capped at (n − 1) by the server."
            />
            <Slider
              label="Clusters (k)"
              value={chosenK}
              min={2}
              max={maxK}
              onChange={setChosenK}
              hint="How many clusters to visualise and sync to Ranking."
            />
          </div>
        </div>

        <div className="mt-6 flex items-center gap-3">
          <button
            onClick={() => fetchCluster()}
            disabled={loading || features.length === 0 || boroughs.length === 0}
            className="btn-primary"
          >
            {loading ? "Running…" : "Run K-Selection"}
          </button>
          {error && <span className="pill-error">{error}</span>}
        </div>
      </SectionCard>

      {result && (
        <>
          <SectionCard
            title="Elbow + Silhouette"
            caption={`Heuristic inertia elbow k = ${result.elbow_k} (grey dashed). Kneedle = ${result.elbow_k_kneedle}. Best silhouette = ${result.best_silhouette_k}. Your partition uses k = ${result.chosen_k}.`}
          >
            {elbowPlot}
          </SectionCard>

          <SectionCard
            title="Cluster count"
            caption="Change k to redraw maps and refresh Ranking labels (same feature set and boroughs)."
            actions={
              <select
                className="glass-select w-auto text-[13px] py-1.5"
                value={chosenK}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setChosenK(v);
                  fetchCluster(v);
                }}
                disabled={loading}
              >
                {result.k_range.map((k) => (
                  <option key={k} value={k}>
                    k = {k}
                  </option>
                ))}
              </select>
            }
          />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectionCard title="Feature Scatter">
              <div className="grid grid-cols-2 gap-2 mb-4">
                <select
                  className="glass-select"
                  value={scatterX}
                  onChange={(e) => setScatterX(e.target.value)}
                >
                  {result.features.map((f) => (
                    <option key={f} value={f}>
                      {formatFeatureLabel(f)}
                    </option>
                  ))}
                </select>
                <select
                  className="glass-select"
                  value={scatterY}
                  onChange={(e) => setScatterY(e.target.value)}
                >
                  {result.features.map((f) => (
                    <option key={f} value={f}>
                      {formatFeatureLabel(f)}
                    </option>
                  ))}
                </select>
              </div>
              {scatterPlot}
            </SectionCard>

            <SectionCard title="Centroid Profiles (z-score)">
              {centroidBars}
            </SectionCard>
          </div>

          {mapPlot && (
            <SectionCard
              title="NYC map"
              caption="CDTA polygons filled by cluster."
            >
              {mapPlot}
            </SectionCard>
          )}

          <SectionCard
            title="Cluster summaries"
            caption="Each card lists the neighborhoods closest to that cluster's centroid in z-space — the most representative members."
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {result.cluster_summaries.map((s) => {
                const closest = (closestByCluster[s.cluster] ?? []).slice(0, 5);
                return (
                  <div key={s.cluster} className="glass-card-tight p-4 flex gap-3">
                    <span
                      aria-hidden
                      className="mt-1 h-3 w-3 rounded-full shrink-0"
                      style={{
                        background: colorForCluster(s.cluster),
                        boxShadow: `0 0 0 3px ${colorForCluster(s.cluster)}22`,
                      }}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] font-medium text-ink">
                        Cluster {s.cluster}
                        <span className="text-muted font-normal"> · n = {s.size}</span>
                      </div>
                      <div className="text-[13px] leading-5 text-muted mt-0.5">
                        {s.description}
                      </div>
                      {closest.length > 0 && (
                        <div className="mt-3">
                          <div className="text-[11px] uppercase tracking-[0.12em] text-muted font-medium mb-1.5">
                            Most representative
                          </div>
                          <ol className="space-y-1 text-[12px] text-ink/85">
                            {closest.map((c, i) => (
                              <li key={c.neighborhood} className="flex items-baseline gap-2">
                                <span className="text-muted tabular-nums w-4 shrink-0">
                                  {i + 1}.
                                </span>
                                <span className="truncate flex-1" title={c.neighborhood}>
                                  {c.neighborhood}
                                </span>
                                <span className="text-muted text-[11px] tabular-nums shrink-0">
                                  d={c.distance.toFixed(2)}
                                </span>
                              </li>
                            ))}
                          </ol>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </SectionCard>

          <SectionCard title="Results table">
            <table className="data-table">
              <thead>
                <tr>
                  <th>k</th>
                  <th>Inertia</th>
                  <th>Silhouette (NumPy)</th>
                  <th>Silhouette (sklearn)</th>
                </tr>
              </thead>
              <tbody>
                {result.k_range.map((k, i) => {
                  const highlight = k === result.chosen_k;
                  return (
                    <tr
                      key={k}
                      className={highlight ? "font-semibold" : ""}
                      style={
                        highlight
                          ? { background: "rgba(251, 191, 36, 0.12)" }
                          : undefined
                      }
                    >
                      <td>{k}</td>
                      <td className="tabular-nums">{result.inertias[i].toFixed(2)}</td>
                      <td className="tabular-nums">{result.silhouettes_numpy[i].toFixed(4)}</td>
                      <td className="tabular-nums">{result.silhouettes_sklearn[i].toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </SectionCard>
        </>
      )}
    </div>
  );
}
