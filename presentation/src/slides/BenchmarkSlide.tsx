import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { CountUp } from '../components/CountUp';
import { StatCard } from '../components/StatCard';
import { headline } from '../data/results';
import { motion as motionTokens } from '../design/theme';

const families = [
  { name: 'TSFMs', count: 4, list: 'Chronos-Bolt · TimesFM 2.5 · Moirai 1.1-R · Time-MoE-200M' },
  { name: 'Classical', count: 2, list: 'MSTL+ETS · SARIMAX (1,0,1)(0,1,1,24)' },
  { name: 'Tabular ML', count: 1, list: 'LightGBM, 50-trial Optuna, 5-seed ensemble' },
  { name: 'Deep (from scratch)', count: 1, list: 'PatchTSMixer, cross-channel-mix mode' },
  { name: 'Composites', count: 3, list: 'ensemble · routed best · meta-router v1/v2' },
];

const harness = [
  ['95% CI', 'Politis–Romano stationary block bootstrap, mean block-length 24 h, 1 000 resamples.'],
  ['DM test', 'Diebold–Mariano with Newey–West HAC variance at truncation lag h−1 = 23.'],
  ['Multiple-comparison', 'Holm sequentially-rejective procedure within each regime, α = 0.05.'],
  ['Reproducibility', '152 pytest checks pin the dataset, parquet schemas, and statistical artefacts.'],
];

export function BenchmarkSlide() {
  return (
    <Slide
      eyebrow="The benchmark"
      title={<>31 systems, one test window, one harness.</>}
      subtitle="Every model writes to the same parquet schema; one statistical pipeline evaluates them all on a held-out 2024–2025 test window. No leakage, no apples-to-oranges."
    >
      {/* Top: four headline metric tiles */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-g80 border-subtle">
        <StatCard label="Systems benchmarked" delay={0.05}>
          <CountUp value={headline.numSystems} className="type-display-02" />
        </StatCard>
        <StatCard label="Test hours (n)" delay={0.1}>
          <CountUp value={headline.testHours} className="type-display-02" />
        </StatCard>
        <StatCard label="DM-test pairs / regime" delay={0.15}>
          <CountUp value={headline.dmPairs} className="type-display-02" />
        </StatCard>
        <StatCard label="Automated pytest checks" delay={0.2}>
          <CountUp value={headline.numTests} className="type-display-02" />
        </StatCard>
      </div>

      {/* Mid: family table */}
      <div className="mt-8">
        <div className="type-label-01 uppercase text-g40 mb-3">Model families</div>
        <div className="border-t border-g80">
          {families.map((f, i) => (
            <motion.div
              key={f.name}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{
                duration: motionTokens.durationModerate02,
                delay: 0.25 + i * 0.05,
              }}
              className="border-b border-g80 py-3 grid grid-cols-12 gap-4 row-hover"
            >
              <div className="col-span-1 type-code-01 text-g40 tabular">
                {String(f.count).padStart(2, '0')}
              </div>
              <div className="col-span-3 type-heading-02 text-g10">{f.name}</div>
              <div className="col-span-8 type-body-01 text-g30">{f.list}</div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Bottom: harness contract */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: motionTokens.durationSlow01, delay: 0.6 }}
        className="mt-6 grid grid-cols-1 md:grid-cols-4 gap-px bg-g80 border-subtle"
      >
        {harness.map(([k, v]) => (
          <div key={k} className="layer-01 p-4">
            <div className="type-label-01 uppercase text-blue40">{k}</div>
            <div className="mt-2 type-body-01 text-g30 leading-snug">{v}</div>
          </div>
        ))}
      </motion.div>
    </Slide>
  );
}
