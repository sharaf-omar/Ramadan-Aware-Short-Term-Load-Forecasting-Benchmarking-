import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { PerRegimeBars, RegimeLegend } from '../components/charts/PerRegimeBars';
import { motion as motionTokens } from '../design/theme';

const winners = [
  { regime: 'Normal',    winner: 'ensemble-top4-residual', mae: 775.1 },
  { regime: 'Ramadan',   winner: 'LightGBM-hijri',         mae: 799.9 },
  { regime: 'Heat-wave', winner: 'Chronos-Bolt L = 720',   mae: 1206.0 },
];

export function PerRegimeSlide() {
  return (
    <Slide
      eyebrow="Per-regime decomposition"
      title={<>Different model families win different regimes.</>}
      subtitle="No single model wins all three. The composite wins by pooling regime specialists — exactly the proposal’s central hypothesis, confirmed empirically."
    >
      <div className="grid grid-cols-12 gap-x-8 mt-2">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.15,
          }}
          className="col-span-12 lg:col-span-8 layer-01 border-subtle p-4 min-h-[440px] flex flex-col"
        >
          <div className="flex-1 min-h-0">
            <PerRegimeBars />
          </div>
          <RegimeLegend />
        </motion.div>

        <div className="col-span-12 lg:col-span-4">
          <div className="type-label-01 uppercase text-g40 mb-3">
            Regime winners
          </div>
          <div className="border-t border-g80">
            {winners.map((w, i) => (
              <motion.div
                key={w.regime}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.3 + i * 0.1,
                }}
                className="border-b border-g80 py-4 grid grid-cols-12 gap-3 items-baseline"
              >
                <div className="col-span-4 type-code-01 uppercase text-blue40">
                  {w.regime}
                </div>
                <div className="col-span-5 type-body-01 text-g10">
                  {w.winner}
                </div>
                <div className="col-span-3 type-heading-03 text-g10 tabular text-right">
                  {w.mae.toFixed(1)}
                </div>
              </motion.div>
            ))}
          </div>
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.7 }}
            className="mt-4 type-helper-01 text-g40 leading-relaxed"
          >
            Compound (Ramadan × Heat-wave) is structurally empty on the
            2018–2025 window; the constraint relaxes around 2030 as the lunar
            calendar drifts later in the Gregorian year.
          </motion.p>
        </div>
      </div>
    </Slide>
  );
}
