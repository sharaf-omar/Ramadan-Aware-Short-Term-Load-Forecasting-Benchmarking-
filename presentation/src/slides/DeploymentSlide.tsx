import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { motion as motionTokens } from '../design/theme';

const options = [
  {
    label: 'Best accuracy',
    mae: 838.8,
    name: 'meta-router-v2',
    body:
      'Four model inferences per τ (Chronos, Time-MoE, LightGBM, Moirai), each residual-corrected, median-aggregated per regime.',
  },
  {
    label: 'Best single-call latency',
    mae: 916.0,
    name: 'routed best-per-regime',
    body:
      'A cheap regime classifier per τ, then exactly one model called downstream. Lowest inference cost in the top tier.',
  },
  {
    label: 'GPU-free',
    mae: 940.4,
    name: 'LightGBM-hijri + residual head',
    body:
      'No GPU at all. The strongest single-architecture system. Drop-in upgrade for an operator already running tabular LightGBM.',
  },
  {
    label: 'Single-GPU TSFM',
    mae: 948.5,
    name: 'Chronos-Bolt L = 720 + residual head',
    body:
      'One GPU model, one CPU residual head. Cleanest end-to-end deployment for teams already on a single TSFM.',
  },
  {
    label: 'Proposal-faithful zero-shot',
    mae: 968.9,
    name: 'Chronos-Bolt L = 720, bare',
    body:
      'No training of any kind, no feature engineering. Accuracy floor for the cheapest possible operational setup.',
  },
];

export function DeploymentSlide() {
  return (
    <Slide
      eyebrow="Deployment recommendations"
      title={<>Five operating points, picked by latency and ops cost.</>}
      subtitle="For short-horizon (1–6 h) sub-day forecasts, ship Time-MoE-200M L = 720 regardless of the headline choice — its h = 1 MAE of 262 is roughly four times better than the next-best model at that horizon."
    >
      <div className="border-t border-g80 mt-2">
        {options.map((o, i) => (
          <motion.div
            key={o.label}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{
              duration: motionTokens.durationModerate02,
              delay: 0.1 + i * 0.07,
            }}
            className="border-b border-g80 py-5 grid grid-cols-12 gap-x-4 row-hover"
          >
            <div className="col-span-12 md:col-span-3">
              <div className="type-code-01 uppercase text-blue40 mb-1.5">
                {o.label}
              </div>
              <div className="type-display-01 text-g10 tabular leading-none">
                {o.mae.toFixed(1)}
              </div>
              <div className="type-label-01 uppercase text-g40 mt-1">MAE (MW)</div>
            </div>
            <div className="col-span-12 md:col-span-9">
              <div className="type-heading-03 text-g10">{o.name}</div>
              <p className="mt-2 type-body-01 text-g30 leading-relaxed max-w-[80ch]">
                {o.body}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </Slide>
  );
}
