"use client";

interface Props {
  label: string;
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
  format?: (s: string) => string;
}

export function MultiSelect({
  label,
  options,
  value,
  onChange,
  format,
}: Props) {
  const toggle = (opt: string) => {
    if (value.includes(opt)) onChange(value.filter((o) => o !== opt));
    else onChange([...value, opt]);
  };
  return (
    <div>
      <div className="text-sm font-medium mb-1">{label}</div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => {
          const active = value.includes(o);
          return (
            <button
              key={o}
              type="button"
              onClick={() => toggle(o)}
              className={
                "text-xs px-2 py-1 rounded border transition " +
                (active
                  ? "bg-ink text-white border-ink"
                  : "bg-white text-ink border-slate-300 hover:border-slate-500")
              }
            >
              {format ? format(o) : o}
            </button>
          );
        })}
      </div>
    </div>
  );
}
