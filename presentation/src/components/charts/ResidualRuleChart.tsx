import {
  ResponsiveContainer,
  ComposedChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Label,
  ReferenceLine,
  Cell,
} from 'recharts';
import { residualImpact } from '../../data/results';
import { carbon, familyColor } from '../../design/theme';
import { GlassTooltip } from './Tooltip';

/**
 * Plan-6 monotonic rule scatter.
 *
 * Per-point label offsets are hand-tuned to fan out the bottom-left
 * cluster (the four strong bare models whose improvement % values
 * are within ~3 percentage points of each other).
 *
 * Labels are drawn as a sibling <Scatter> overlay using a custom shape
 * that renders only <text>, no circles. Recharts' built-in <LabelList>
 * has no per-point anchor/offset control, so this is the cleanest way
 * to fan the cluster labels without overlap.
 */
const labelOffset: Record<
  string,
  { dx: number; dy: number; anchor: 'start' | 'end' }
> = {
  'SARIMAX-hijri':       { dx:  12, dy:   4, anchor: 'start' },
  'PatchTSMixer L=168':  { dx:  12, dy:   4, anchor: 'start' },
  'Moirai L=336':        { dx:  12, dy:   4, anchor: 'start' },
  'MSTL+ETS-hijri':      { dx:  12, dy:  -8, anchor: 'start' },
  'TimesFM L=168':       { dx: -10, dy:  -8, anchor: 'end'   },
  'LightGBM-nohijri':    { dx:  12, dy: -10, anchor: 'start' },
  'LightGBM-hijri':      { dx:  12, dy:   8, anchor: 'start' },
  'Time-MoE L=720':      { dx:  12, dy:  22, anchor: 'start' },
  'Chronos-Bolt L=720':  { dx: -10, dy:  22, anchor: 'end'   },
};

export function ResidualRuleChart() {
  const data = residualImpact.map((p) => ({
    name: p.model,
    bareMAE: p.bareMAE,
    improvement: -p.deltaPct,
    family: p.family,
  }));

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart
        data={data}
        margin={{ top: 28, right: 56, bottom: 56, left: 64 }}
      >
        <CartesianGrid stroke={carbon.borderSubtle} strokeDasharray="0" />
        <XAxis
          type="number"
          dataKey="bareMAE"
          domain={[900, 2600]}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
        >
          <Label
            value="Bare-model MAE (MW)  →  weaker"
            position="insideBottom"
            offset={-24}
            fill={carbon.textHelper}
            style={{ fontFamily: 'IBM Plex Mono', fontSize: 11 }}
          />
        </XAxis>
        <YAxis
          type="number"
          dataKey="improvement"
          domain={[0, 50]}
          tickFormatter={(v) => `${v}%`}
          tick={{ fill: carbon.textSecondary }}
          stroke={carbon.borderSubtle}
        >
          <Label
            value="Improvement from residual head"
            angle={-90}
            position="insideLeft"
            offset={-18}
            fill={carbon.textHelper}
            style={{ fontFamily: 'IBM Plex Mono', fontSize: 11 }}
          />
        </YAxis>
        <Tooltip
          cursor={{ stroke: carbon.blue60, strokeOpacity: 0.4, strokeWidth: 1 }}
          content={<GlassTooltip />}
        />

        {/* "Meaningful rescue" reference line */}
        <ReferenceLine
          y={10}
          stroke={carbon.borderSubtle}
          strokeDasharray="2 4"
          label={{
            value: '10% threshold',
            position: 'insideTopRight',
            fill: carbon.textHelper,
            fontSize: 10,
            fontFamily: 'IBM Plex Mono',
          }}
        />

        {/* Points — plain Scatter with Cell-per-point colors */}
        <Scatter
          name="residual lift"
          dataKey="improvement"
          isAnimationActive
          animationDuration={1400}
          animationEasing="ease-out"
        >
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={familyColor[d.family as keyof typeof familyColor]}
              stroke={carbon.background}
              strokeWidth={2}
            />
          ))}
        </Scatter>

        {/* Labels — separate Scatter overlay, shape returns only <text> */}
        <Scatter
          dataKey="improvement"
          isAnimationActive
          animationDuration={1600}
          animationBegin={900}
          legendType="none"
          shape={(props: any) => {
            const { cx, cy, payload } = props;
            if (cx == null || cy == null) return <g />;
            const off =
              labelOffset[payload.name] ?? { dx: 12, dy: 4, anchor: 'start' };
            return (
              <g>
                <text
                  x={cx + off.dx}
                  y={cy + off.dy}
                  fill={carbon.textPrimary}
                  fontSize={12}
                  fontFamily="IBM Plex Sans"
                  textAnchor={off.anchor}
                >
                  {payload.name}
                </text>
                <text
                  x={cx + off.dx}
                  y={cy + off.dy + 13}
                  fill={carbon.textHelper}
                  fontSize={11}
                  fontFamily="IBM Plex Mono"
                  textAnchor={off.anchor}
                >
                  −{payload.improvement.toFixed(1)}%
                </text>
              </g>
            );
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
