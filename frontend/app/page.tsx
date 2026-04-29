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

const CANDIDATE_FEATURES = [
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

const DEFAULT_FEATURES = [
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

export default function KSelectionPage() {
  const [ranges, setRanges] = useState<FeatureRangesResponse | null>(null);
  const [boroughs, setBoroughs] = useState<string[]>([]);
  const [features, setFeatures] = useState<string[]>(DEFAULT_FEATURES);
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
          scatterX +
          ": %{x:.1f}<br>" +
          scatterY +
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
          xaxis: { title: { text: scatterX } },
          yaxis: { title: { text: scatterY } },
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
          x: result.features,
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
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-ink">K-Selection / Clustering</h1>
        <p className="text-sm text-muted mt-1">
          Sweep k = 2…max_k for elbow charts; pick <strong>Clusters (k)</strong>{" "}
          for the partition shown on maps and synced to Ranking.
        </p>
      </div>

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
              options={CANDIDATE_FEATURES}
              value={features}
              onChange={setFeatures}
              format={(f) => f.replace(/_/g, " ")}
            />
            <Slider
              label="Maximum k"
              value={maxK}
              min={3}
              max={15}
              onChange={(v) => {
                setMaxK(v);
                setChosenK(v);
              }}
              hint="Upper bound for k sweep. Capped at (n − 1) by the server."
            />
            <Slider
              label="Clusters (k)"
              value={chosenK}
              min={2}
              max={maxK}
              onChange={setChosenK}
              hint="How many clusters to visualize and sync to Ranking (must be within the sweep)."
            />
          </div>
        </div>

        <div className="mt-6 flex items-center gap-3">
          <button
            onClick={() => fetchCluster()}
            disabled={loading || features.length === 0 || boroughs.length === 0}
            className="bg-ink text-white px-4 py-2 rounded text-sm font-medium disabled:opacity-50"
          >
            {loading ? "Running…" : "Run K-Selection Analysis"}
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
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

          <SectionCard title="Cluster count">
            <p className="text-sm text-muted mb-2">
              Change k and apply to redraw maps and refresh Ranking labels (same
              feature set and boroughs).
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <label className="text-sm font-medium">
                Clusters (k){" "}
                <select
                  className="border rounded px-2 py-1 text-sm ml-2"
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
                      {k}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </SectionCard>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SectionCard title="Feature Scatter">
              <div className="grid grid-cols-2 gap-2 mb-3">
                <select
                  className="border rounded px-2 py-1 text-sm"
                  value={scatterX}
                  onChange={(e) => setScatterX(e.target.value)}
                >
                  {result.features.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
                <select
                  className="border rounded px-2 py-1 text-sm"
                  value={scatterY}
                  onChange={(e) => setScatterY(e.target.value)}
                >
                  {result.features.map((f) => (
                    <option key={f} value={f}>
                      {f}
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

          <SectionCard title="Cluster summaries">
            <div className="space-y-3">
              {result.cluster_summaries.map((s) => (
                <div
                  key={s.cluster}
                  className="border-l-4 pl-3 py-1"
                  style={{ borderColor: colorForCluster(s.cluster) }}
                >
                  <div className="font-medium text-sm">
                    Cluster {s.cluster} · n = {s.size}
                  </div>
                  <div className="text-sm text-muted">{s.description}</div>
                </div>
              ))}
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
                      style={
                        highlight
                          ? {
                              background: "#fef9c3",
                              fontWeight: 600,
                            }
                          : undefined
                      }
                    >
                      <td>{k}</td>
                      <td>{result.inertias[i].toFixed(2)}</td>
                      <td>{result.silhouettes_numpy[i].toFixed(4)}</td>
                      <td>{result.silhouettes_sklearn[i].toFixed(4)}</td>
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
