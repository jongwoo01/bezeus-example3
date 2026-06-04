type StatusMetricProps = {
  label: string;
  value: string;
  emphasized?: boolean;
};

export function StatusMetric({ label, value, emphasized = false }: StatusMetricProps) {
  return (
    <div className="flex justify-between gap-4 border-b border-white/10 pb-3 last:border-b-0 last:pb-0">
      <dt className="text-white/45">{label}</dt>
      <dd className={emphasized ? "font-medium text-cyan-100" : "text-white/80"}>{value}</dd>
    </div>
  );
}
