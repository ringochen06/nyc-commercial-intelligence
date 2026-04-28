import type {
  CdtaGeoResponse,
  ClusterResponse,
  FeatureRangesResponse,
  HealthResponse,
  RankRequest,
  RankResponse,
  Vintage,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    let detail = text;
    try {
      const j = JSON.parse(text);
      detail = j.detail || j.message || text;
    } catch {}
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  featureRanges: (vintage: Vintage = "present") =>
    request<FeatureRangesResponse>(`/api/feature-ranges?vintage=${vintage}`),
  cdtaGeo: () => request<CdtaGeoResponse>("/api/geo/cdta"),
  cluster: (body: {
    features: string[];
    boroughs?: string[];
    max_k: number;
    vintage: Vintage;
    random_state?: number;
  }) =>
    request<ClusterResponse>("/api/cluster", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  rank: (body: RankRequest) =>
    request<RankResponse>("/api/rank", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  agent: (body: RankRequest) =>
    request<{ answer: string; n_filtered: number }>("/api/agent", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};

export { API_BASE };
