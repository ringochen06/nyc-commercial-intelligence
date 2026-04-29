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
      <div className="text-[13px] font-medium text-ink mb-2">{label}</div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((o) => {
          const active = value.includes(o);
          return (
            <button
              key={o}
              type="button"
              onClick={() => toggle(o)}
              className={`pill-chip ${active ? "is-active" : ""}`}
            >
              {format ? format(o) : o}
            </button>
          );
        })}
      </div>
    </div>
  );
}
