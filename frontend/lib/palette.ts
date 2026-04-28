export const CLUSTER_PALETTE: string[] = [
  "#4A90D9",
  "#E74C3C",
  "#2ECC71",
  "#F39C12",
  "#9B59B6",
  "#1ABC9C",
  "#E67E22",
  "#3498DB",
  "#E91E63",
  "#00BCD4",
  "#8BC34A",
  "#FF5722",
  "#795548",
  "#607D8B",
  "#FF9800",
  "#673AB7",
  "#009688",
  "#F44336",
  "#CDDC39",
  "#03A9F4",
];

export const colorForCluster = (c: number): string =>
  CLUSTER_PALETTE[c % CLUSTER_PALETTE.length];
