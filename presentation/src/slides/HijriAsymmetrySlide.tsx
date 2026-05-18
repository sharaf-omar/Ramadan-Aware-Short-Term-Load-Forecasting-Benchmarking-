import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { HijriDeltaChart } from '../components/charts/HijriDeltaChart';
import { motion as motionTokens } from '../design/theme';

const mechanisms = [
  {
    tag: 'Why HF API hurts',
    body:
      'Auxiliary channels append into a pretrained attention context that has no prior over them. The features function as distractors and degrade attention precision.',
  },
  {
    tag: 'Why residual head helps',
    body:
      'The head learns the calendar effect on the residual signal and adds it back as a separate correction stage that the foundation model never sees.',
  },
  {
    tag: 'The general rule',
    body:
      'Domain-specific covariates whose distribution is unlike anything in the pretraining corpus should be external corrections — not internal model inputs.',
  },
];

export function HijriAsymmetrySlide() {
  return (
    <Slide
      eyebrow="Ablation A · the central finding"
      title={<>Same features. Help as a residual head, hurt as a covariate.</>}
      subtitle="The same is_ramadan / day_of_ramadan / is_eid features change a model’s Ramadan MAE by ±25% depending entirely on how they are injected. This is the report’s most theoretically interesting result."
    >
      <div className="grid grid-cols-12 gap-x-8 mt-2">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.15,
          }}
          className="col-span-12 lg:col-span-8 layer-01 border-subtle p-4 min-h-[460px]"
        >
          <HijriDeltaChart />
        </motion.div>

        <div className="col-span-12 lg:col-span-4">
          <div className="type-label-01 uppercase text-g40 mb-3">Mechanism</div>
          <div className="border-t border-g80">
            {mechanisms.map((m, i) => (
              <motion.div
                key={m.tag}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.3 + i * 0.1,
                }}
                className="border-b border-g80 py-4"
              >
                <div className="type-code-01 uppercase text-blue40 mb-1.5">
                  {m.tag}
                </div>
                <div className="type-body-01 text-g30 leading-relaxed">
                  {m.body}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </Slide>
  );
}
