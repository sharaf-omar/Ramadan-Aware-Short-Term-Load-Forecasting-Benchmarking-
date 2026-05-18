import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { EmphasisBar } from '../components/EmphasisBar';
import Stepper, { StepperItem } from '../components/reactbits/Stepper';
import { motion as motionTokens } from '../design/theme';

const stages: StepperItem[] = [
  {
    label: 'Chronos-Bolt L = 720, bare',
    metric: '968.9',
    description: 'strongest single bare model',
  },
  {
    label: 'LightGBM-hijri + residual',
    metric: '940.4',
    description: 'strongest single model overall',
  },
  {
    label: 'routed best-per-regime',
    metric: '916.0',
    description: 'one specialist per regime',
  },
  {
    label: 'ensemble of 4 (mixed)',
    metric: '891.4',
    description: 'median of 4 strong models',
  },
  {
    label: 'ensemble of 4 residual-corrected',
    metric: '872.4',
    description: 'median of 4 residual versions',
  },
  {
    label: 'meta-router v1',
    metric: '840.9',
    description: 'ensemble Normal + LGBM Ramadan + Chronos Heat-wave',
  },
  {
    label: 'meta-router v2',
    metric: '838.8',
    description: 'v1 with 4-model Heat-wave ensemble',
    highlight: true,
  },
];

export function CompositesSlide() {
  return (
    <Slide
      eyebrow="Composite construction"
      title={
        <>
          From <span className="tabular">968.9</span> to{' '}
          <span className="text-blue40 tabular">838.8</span>, one step at a time.
        </>
      }
      subtitle="Each composite stage exposes a different lever. About 60% of the headline gain comes from per-regime specialisation; the other 40% from ensembling specialists where their CIs overlap."
    >
      <div className="mt-2">
        <Stepper items={stages} startDelay={0.15} stagger={motionTokens.cinemaStagger} />

        {/* Bottom emphasis bar showing total ladder span */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationModerate02,
            delay: 0.15 + stages.length * motionTokens.cinemaStagger + 0.3,
          }}
          className="mt-4"
        >
          <EmphasisBar
            widthPct={100}
            color="bg-blue60"
            delay={0.15 + stages.length * motionTokens.cinemaStagger + 0.4}
            duration={1.0}
          />
        </motion.div>
      </div>
    </Slide>
  );
}
