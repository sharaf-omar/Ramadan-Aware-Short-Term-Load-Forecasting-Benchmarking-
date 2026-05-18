import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { lSweep } from '../../data/results';
import { carbon, categorical } from '../../design/theme';
import { GlassTooltip } from './Tooltip';

const series = [
  { key: 'Chronos', color: categorical[1] }, // cyan-50
  { key: 'TimesFM', color: categorical[0] }, // purple-70
  { key: 'Moirai',  color: categorical[3] }, // magenta-70
  { key: 'TimeMoE', color: categorical[6] }, // green-60
];

export function LSweepChart() {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart
        data={lSweep}
        margin={{ top: 16, right: 24, bottom: 32, left: 24 }}
      >
        <CartesianGrid stroke={carbon.borderSubtle} strokeDasharray="0" />
        <XAxis
          dataKey="L"
          type="number"
          domain={[80, 760]}
          ticks={[96, 168, 336, 720]}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
          label={{
            value: 'context length L (hours)',
            position: 'insideBottom',
            offset: -16,
            fill: carbon.textHelper,
            fontSize: 11,
            fontFamily: 'IBM Plex Mono',
          }}
        />
        <YAxis
          domain={[900, 2000]}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
          label={{
            value: 'aggregate MAE (MW)',
            angle: -90,
            position: 'insideLeft',
            fill: carbon.textHelper,
            fontSize: 11,
            fontFamily: 'IBM Plex Mono',
          }}
        />
        <Tooltip content={<GlassTooltip />} />
        <Legend
          verticalAlign="top"
          height={28}
          iconType="plainline"
          wrapperStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 12 }}
        />
        {series.map(({ key, color }, i) => (
          <Line
            key={key}
            type="monotone"
            dataKey={key}
            stroke={color}
            strokeWidth={2}
            dot={{ r: 4, strokeWidth: 1, fill: carbon.background, stroke: color }}
            activeDot={{ r: 6 }}
            isAnimationActive
            animationDuration={1600}
            animationEasing="ease-out"
            animationBegin={i * 220}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
