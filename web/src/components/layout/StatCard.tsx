interface StatCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  color?: string;
}

export default function StatCard({ label, value, subtitle, color = 'var(--accent)' }: StatCardProps) {
  return (
    <div className="bg-[var(--surface)] rounded-xl border border-[var(--border)] p-5">
      <p className="text-xs text-[var(--text-secondary)] uppercase tracking-wider">{label}</p>
      <p className="text-2xl font-bold mt-1" style={{ color }}>{value}</p>
      {subtitle && <p className="text-xs text-[var(--text-secondary)] mt-1">{subtitle}</p>}
    </div>
  );
}
