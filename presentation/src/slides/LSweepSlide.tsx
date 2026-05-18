import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { LSweepChart } from '../components/charts/LSweepChart';
import { motion as motionTokens } from '../design/theme';

const responses = [
  {
    pattern: 'Monotone',
    models: 'Chronos-Bolt-Base · Time-MoE-200M',
    body: 'More history reliably helps. Best L = 720.',
  },
  {
    pattern: 'Non-monotone (TimesFM)',
    models: 'TimesFM 2.5',
    body: 'Best L = 168. Longer context actively hurts; L = 336 regresses by +202 MAE.',
  },
  {
    pattern: 'Non-monotone (Moirai)',
    models: 'Moirai 1.1-R-small',
    body: 'Best L = 336. Recovers some accuracy past the L = 168 dip but never below L = 336.',
  },
];

export function LSweepSlide() {
  return (
    <Slide
      eyebrow="Ablation C · context-length sensitivity"
      title={<>Two response patterns: monotone and non-monotone.</>}
      subtitle="Chronos and Time-MoE benefit from longer context all the way to L = 720. TimesFM peaks at L = 168 and regresses afterward — likely a positional-encoding bias inherited from its pretraining curriculum."
    >
      <div className="grid grid-cols-12 gap-x-8 mt-2">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: motionTokens.durationSlow01, delay: 0.15 }}
          className="col-span-12 lg:col-span-8 layer-01 border-subtle p-4 min-h-[440px]"
        >
          <LSweepChart />
        </motion.div>

        <div className="col-span-12 lg:col-span-4">
          <div className="type-label-01 uppercase text-g40 mb-3">Response patterns</div>
          <div className="border-t border-g80">
            {responses.map((r, i) => (
              <motion.div
                key={r.pattern}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.3 + i * 0.1,
                }}
                className="border-b border-g80 py-4"
              >
                <div className="type-code-01 uppercase text-blue40 mb-1">
                  {r.pattern}
                </div>
                <div className="type-heading-02 text-g10 mb-1">{r.models}</div>
                <div className="type-body-01 text-g30 leading-snug">
                  {r.body}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </Slide>
  );
}
