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
  ErrorBar,
} from 'recharts';
import { leaderboard } from '../../data/results';
import { carbon, familyColor } from '../../design/theme';
import { GlassTooltip } from './Tooltip';

/**
 * Horizontal forest plot of top-15 systems with 95% bootstrap CI whiskers.
 * Carbon styling: sharp bars (no border-radius), neutral grid, Plex Mono axes.
 */
export function LeaderboardChart() {
  const data = leaderboard.map((r) => ({
    name: r.shortName,
    rank: r.rank,
    mae: r.mae,
    family: r.family,
    err: [r.mae - r.ciLow, r.ciHigh - r.mae] as [number, number],
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 8, right: 56, bottom: 8, left: 8 }}
        barCategoryGap={3}
      >
        <CartesianGrid stroke={carbon.borderSubtle} strokeDasharray="0" horizontal={false} />
        <XAxis
          type="number"
          domain={[700, 1300]}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
        />
        <YAxis
          type="category"
          dataKey="name"
          width={180}
          tick={{ fill: carbon.textPrimary, fontSize: 12, fontFamily: 'IBM Plex Sans' }}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          cursor={{ fill: carbon.layer02 }}
          content={<GlassTooltip />}
        />
        <ReferenceLine
          x={979.0}
          stroke={carbon.borderStrong}
          strokeDasharray="4 2"
          label={{
            value: 'LGBM-hijri baseline',
            position: 'insideTopRight',
            fill: carbon.textHelper,
            fontSize: 11,
            fontFamily: 'IBM Plex Mono',
          }}
        />
        <Bar
          dataKey="mae"
          isAnimationActive
          animationDuration={1400}
          animationEasing="ease-out"
        >
          {data.map((d, i) => (
            <Cell key={i} fill={familyColor[d.family]} />
          ))}
          <ErrorBar
            dataKey="err"
            width={5}
            stroke={carbon.textSecondary}
            strokeOpacity={0.65}
            direction="x"
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
