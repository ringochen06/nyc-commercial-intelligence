"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface ClusterShare {
  // neighborhood -> cluster id, populated when K-Selection runs.
  assignments: Record<string, number>;
  // cluster id (as string) -> short brief.
  briefs: Record<string, string>;
  k: number | null;
  features: string[];
  setClustering: (
    assignments: Record<string, number>,
    briefs: Record<string, string>,
    k: number,
    features: string[],
  ) => void;
  clear: () => void;
}

export const useClusterStore = create<ClusterShare>()(
  persist(
    (set) => ({
      assignments: {},
      briefs: {},
      k: null,
      features: [],
      setClustering: (assignments, briefs, k, features) =>
        set({ assignments, briefs, k, features }),
      clear: () => set({ assignments: {}, briefs: {}, k: null, features: [] }),
    }),
    { name: "nyc-cluster-share" },
  ),
);
