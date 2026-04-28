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

export function PlotlyChart({ data, layout, style, className }: Props) {
  return (
    <Plot
      data={data}
      layout={{
        autosize: true,
        margin: { l: 50, r: 30, t: 30, b: 50 },
        paper_bgcolor: "white",
        plot_bgcolor: "white",
        ...(layout || {}),
      }}
      useResizeHandler
      style={{ width: "100%", height: "420px", ...(style || {}) }}
      config={{ displayModeBar: false, responsive: true }}
      className={className}
    />
  );
}
