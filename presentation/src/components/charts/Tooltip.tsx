import { TooltipProps } from 'recharts';

/**
 * Carbon-style chart tooltip: solid layer-02 surface, sharp corners,
 * 1px subtle border, no blur or glow. Numeric values use Plex Mono.
 */
export function GlassTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="layer-02 border border-g70 px-3 py-2 shadow-none">
      {label !== undefined && (
        <div className="type-code-01 uppercase text-g40 mb-1">{label}</div>
      )}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 type-body-01">
          <span
            className="inline-block w-2.5 h-2.5"
            style={{ background: p.color }}
          />
          <span className="text-g10">{p.name}</span>
          <span className="tabular text-g30 ml-auto">
            {typeof p.value === 'number' ? p.value.toFixed(1) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}
