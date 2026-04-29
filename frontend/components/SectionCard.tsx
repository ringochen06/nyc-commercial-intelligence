interface Props {
  title?: React.ReactNode;
  caption?: React.ReactNode;
  actions?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
}

export function SectionCard({ title, caption, actions, children, className }: Props) {
  const hasHeader = Boolean(title || actions);
  return (
    <section className={`glass-card p-6 ${className ?? ""}`}>
      {hasHeader && (
        <div className="flex items-start justify-between gap-3">
          {title && (
            <h2 className="text-[15px] font-semibold text-ink tracking-tightish">
              {title}
            </h2>
          )}
          {actions && <div className="shrink-0">{actions}</div>}
        </div>
      )}
      {caption && (
        <p className="text-[13px] leading-5 text-muted mt-1">{caption}</p>
      )}
      {children && <div className={hasHeader || caption ? "mt-4" : ""}>{children}</div>}
    </section>
  );
}
