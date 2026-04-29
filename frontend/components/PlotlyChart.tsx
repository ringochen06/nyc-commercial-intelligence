"use client";

import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";

// Dynamic import: Plotly is browser-only; SSR can't run it.
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  data: Data[];
  layout?: Partial<Layout>;
  style?: React.CSSProperties;
  className?: string;
}

const FONT_FAMILY =
  '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", "Segoe UI", Roboto, sans-serif';

export function PlotlyChart({ data, layout, style, className }: Props) {
  return (
    <Plot
      data={data}
      layout={{
        autosize: true,
        margin: { l: 50, r: 30, t: 30, b: 50 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { family: FONT_FAMILY, size: 12, color: "#0f172a" },
        xaxis: {
          gridcolor: "rgba(15,23,42,0.06)",
          zerolinecolor: "rgba(15,23,42,0.12)",
          linecolor: "rgba(15,23,42,0.12)",
          ...(layout?.xaxis || {}),
        },
        yaxis: {
          gridcolor: "rgba(15,23,42,0.06)",
          zerolinecolor: "rgba(15,23,42,0.12)",
          linecolor: "rgba(15,23,42,0.12)",
          ...(layout?.yaxis || {}),
        },
        ...(layout || {}),
      }}
      useResizeHandler
      style={{ width: "100%", height: "420px", ...(style || {}) }}
      config={{ displayModeBar: false, responsive: true }}
      className={className}
    />
  );
}
