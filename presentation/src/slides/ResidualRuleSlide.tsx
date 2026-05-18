import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { Slide } from '../components/Slide';
import { CountUp } from '../components/CountUp';
import { EmphasisBar } from '../components/EmphasisBar';
import { ResidualRuleChart } from '../components/charts/ResidualRuleChart';
import { residualImpact } from '../data/results';
import { motion as motionTokens } from '../design/theme';

const rescues = [
  { model: 'SARIMAX-hijri',         bare: 2485.9, corrected: 1299.3, delta: -47.7, note: 'almost halved' },
  { model: 'PatchTSMixer L = 168',  bare: 1552.7, corrected: 1045.8, delta: -32.6, note: 'one third off' },
  { model: 'Moirai L = 336',        bare: 1727.1, corrected: 1317.2, delta: -23.7, note: 'a quarter rescued' },
] as const;

const notes = [
  ['Monotonic rule',     'Lift scales near-monotonically with bare-model weakness across all nine bases — from −47.7% on SARIMAX down to −2.1% on Chronos.'],
  ['Even the incumbent', 'LightGBM-hijri, already tuned on the v2 features, still gains 4% from a head on its own residuals. Nonlinear interactions the L2 objective missed.'],
  ['Regime routing',     'Train on Normal + Ramadan only; pass Heat-wave τ values to the bare base. Without it, Normal-regime bias regresses Heat-wave 25–32% on the strong TSFMs.'],
];

export function ResidualRuleSlide() {
  const avgImprovement =
    residualImpact.reduce((acc, p) => acc + Math.abs(p.deltaPct), 0) /
    residualImpact.length;

  return (
    <Slide
      eyebrow="Plan 6 · the rescue effect"
      title={
        <>
          One LightGBM head rescues nine base models — up to{' '}
          <span className="text-blue40 tabular">
            −<CountUp value={47.7} decimals={1} />%
          </span>
          .
        </>
      }
      subtitle={
        <>
          A single post-hoc residual head with regime-stratified routing
          improves <em>every</em> base model in the benchmark. Average
          improvement across the nine bases is{' '}
          <span className="text-blue40 tabular">
            −<CountUp value={avgImprovement} decimals={1} />%
          </span>
          .
        </>
      }
    >
      {/* Compact rescue strip — three biggest deltas */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-g80 border-subtle">
        {rescues.map((r, i) => (
          <motion.div
            key={r.model}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: motionTokens.durationModerate02,
              delay: 0.1 + i * 0.06,
              ease: motionTokens.easingEntrance,
            }}
            className="layer-01 p-4"
          >
            <div className="flex items-baseline justify-between">
              <div className="type-code-01 uppercase text-g40">{r.model}</div>
              <div className="type-code-01 uppercase text-blue40">{r.note}</div>
            </div>
            <div className="mt-2 flex items-baseline gap-3">
              <div className="type-display-02 text-g10 tabular leading-none">
                −<CountUp
                  value={Math.abs(r.delta)}
                  decimals={1}
                  duration={1.2}
                />%
              </div>
              <div className="type-code-01 text-g40 tabular">
                {r.bare.toFixed(0)}
                <ArrowRight size={11} className="inline mx-1 text-g60" />
                {r.corrected.toFixed(0)}
              </div>
            </div>
            {/* Carbon-style emphasis: a thin bar drawing in once the count lands */}
            <div className="mt-2">
              <EmphasisBar
                height={2}
                widthPct={Math.min(100, Math.abs(r.delta) * 2)}
                delay={0.2 + i * 0.06 + 1.0}
                duration={0.7}
              />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Full-width chart — takes all available vertical space */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{
          duration: motionTokens.durationSlow01,
          delay: 0.35,
        }}
        className="mt-4 layer-01 border-subtle p-4 h-[460px]"
      >
        <ResidualRuleChart />
      </motion.div>

      {/* Notes as a horizontal strip below the chart */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-px bg-g80 border-subtle">
        {notes.map(([tag, body], i) => (
          <motion.div
            key={tag}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{
              duration: motionTokens.durationModerate02,
              delay: 0.55 + i * 0.07,
            }}
            className="layer-01 px-4 py-3"
          >
            <div className="type-code-01 uppercase text-blue40 mb-1">{tag}</div>
            <div className="type-body-01 text-g30 leading-snug">{body}</div>
          </motion.div>
        ))}
      </div>
    </Slide>
  );
}
