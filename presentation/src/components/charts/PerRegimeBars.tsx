import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import { carbon, regimeColor } from '../../design/theme';
import { GlassTooltip } from './Tooltip';

// Top 8 systems × 3 regimes (Compound is empty).
const data = [
  { system: 'meta-router v2',   Normal: 793.0, Ramadan: 799.9,  'Heat-wave': 1206.0 },
  { system: 'ensemble + res',   Normal: 775.1, Ramadan: 859.4,  'Heat-wave': 1242.6 },
  { system: 'stacked-LGBM',     Normal: 803.4, Ramadan: 893.1,  'Heat-wave': 1267.5 },
  { system: 'routed best',      Normal: 825.7, Ramadan: 799.9,  'Heat-wave': 1206.0 },
  { system: 'LGBM-hijri + res', Normal: 875.6, Ramadan: 815.7,  'Heat-wave': 1239.0 },
  { system: 'Chronos L=720',    Normal: 880.2, Ramadan: 1057.0, 'Heat-wave': 1208.7 },
  { system: 'LGBM-hijri',       Normal: 909.3, Ramadan: 858.4,  'Heat-wave': 1257.9 },
  { system: 'Time-MoE L=720',   Normal: 901.4, Ramadan: 1118.6, 'Heat-wave': 1241.7 },
];

const regimes: Array<keyof typeof regimeColor> = ['Normal', 'Ramadan', 'Heat-wave'];

export function PerRegimeBars() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 20, right: 20, bottom: 36, left: 8 }}>
        <CartesianGrid stroke={carbon.borderSubtle} strokeDasharray="0" vertical={false} />
        <XAxis
          dataKey="system"
          tick={{ fill: carbon.textSecondary, fontSize: 11 }}
          angle={-22}
          textAnchor="end"
          height={70}
          interval={0}
          stroke={carbon.borderSubtle}
        />
        <YAxis
          domain={[700, 1400]}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
          label={{
            value: 'MAE (MW)',
            angle: -90,
            position: 'insideLeft',
            fill: carbon.textHelper,
            fontSize: 11,
            fontFamily: 'IBM Plex Mono',
          }}
        />
        <Tooltip cursor={{ fill: carbon.layer02 }} content={<GlassTooltip />} />
        {regimes.map((r) => (
          <Bar
            key={r}
            dataKey={r}
            isAnimationActive
            animationDuration={1300}
            animationEasing="ease-out"
          >
            {data.map((_, i) => (
              <Cell key={i} fill={regimeColor[r]} />
            ))}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function RegimeLegend() {
  return (
    <div className="flex flex-wrap items-center gap-5 mt-4">
      {regimes.map((r) => (
        <div key={r} className="flex items-center gap-2 type-code-01 text-g30">
          <span
            className="inline-block w-3 h-3"
            style={{ background: regimeColor[r] }}
          />
          {r}
        </div>
      ))}
    </div>
  );
}
