import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Cell,
  Tooltip,
} from 'recharts';
import { hijriDelta } from '../../data/results';
import { carbon } from '../../design/theme';
import { GlassTooltip } from './Tooltip';

// Semantic colors only: green for "helps", red for "hurts", gray for "no effect".
const injColor: Record<string, string> = {
  'residual head':       carbon.green40,
  'feature engineering': carbon.green40,
  'cross-channel mix':   carbon.textHelper,
  'HF covariate API':    carbon.red50,
};

export function HijriDeltaChart() {
  const data = [...hijriDelta]
    .sort((a, b) => a.deltaPct - b.deltaPct)
    .map((d) => ({
      label: `${d.model} (${d.injection})`,
      delta: d.deltaPct,
      injection: d.injection,
    }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 8, right: 56, bottom: 8, left: 8 }}
        barCategoryGap={4}
      >
        <CartesianGrid stroke={carbon.borderSubtle} strokeDasharray="0" horizontal={false} />
        <XAxis
          type="number"
          domain={[-32, 32]}
          tickFormatter={(v) => `${v > 0 ? '+' : ''}${v}%`}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
        />
        <YAxis
          type="category"
          dataKey="label"
          width={280}
          tick={{ fill: carbon.textPrimary, fontSize: 11, fontFamily: 'IBM Plex Sans' }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip cursor={{ fill: carbon.layer02 }} content={<GlassTooltip />} />
        <ReferenceLine x={0} stroke={carbon.borderStrong} strokeWidth={1} />
        <Bar
          dataKey="delta"
          isAnimationActive
          animationDuration={1400}
          animationEasing="ease-out"
        >
          {data.map((d, i) => (
            <Cell key={i} fill={injColor[d.injection] ?? carbon.textHelper} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
