"use client";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  hint?: string;
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  hint,
}: Props) {
  // Clamp incoming value to [min, max] so a stale parent state doesn't
  // pin the thumb beyond the slider track (browsers refuse to render
  // out-of-range values, which makes the slider look frozen).
  const safeValue = Math.min(max, Math.max(min, value));
  const display = format ? format(safeValue) : safeValue.toLocaleString();
  const span = max - min;
  const ratio = span > 0 ? (safeValue - min) / span : 0;
  const fill = `${(ratio * 100).toFixed(2)}%`;
  return (
    <label className="block">
      <div className="flex justify-between items-baseline">
        <span className="text-[13px] font-medium text-ink">{label}</span>
        <span className="text-[12px] text-muted tabular-nums">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={safeValue}
        onChange={(e) => onChange(Number(e.target.value))}
        className="glass-range mt-1.5"
        style={{ ["--fill" as string]: fill }}
      />
      {hint && <p className="text-[11px] text-muted mt-1 leading-4">{hint}</p>}
    </label>
  );
}
