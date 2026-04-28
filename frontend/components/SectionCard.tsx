interface Props {
  title?: React.ReactNode;
  caption?: React.ReactNode;
  children: React.ReactNode;
}

export function SectionCard({ title, caption, children }: Props) {
  return (
    <section className="border border-slate-200 rounded-lg p-4 bg-white shadow-sm">
      {title && (
        <h2 className="text-lg font-semibold text-ink mb-1">{title}</h2>
      )}
      {caption && <p className="text-sm text-muted mb-3">{caption}</p>}
      {children}
    </section>
  );
}
