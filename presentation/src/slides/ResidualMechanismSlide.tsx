import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { Slide } from '../components/Slide';
import { motion as motionTokens } from '../design/theme';

// Pipeline stages — left to right, with arrows between.
interface PipelineStage {
  no: string;
  tag: string;
  title: string;
  body: string;
  accent?: boolean;
}

const stages: PipelineStage[] = [
  {
    no: '01',
    tag: 'Inputs',
    title: 'Features x_τ',
    body: 'Weather, lag, calendar, and (for the hijri variant) Hijri + Ramadan × hour interactions.',
  },
  {
    no: '02',
    tag: 'Base model',
    title: 'Bare prediction ŷ_base',
    body: 'Any of nine base models — TSFM, LightGBM, MSTL+ETS, SARIMAX, or PatchTSMixer.',
  },
  {
    no: '03',
    tag: 'Residual head',
    title: 'r̂ = LGBM(x_τ)',
    body: 'A second-stage LightGBM trained on the L1 residual target (y − ŷ_base), seeing the same x_τ.',
  },
  {
    no: '04',
    tag: 'Router',
    title: 'ŷ_final',
    body: 'Normal / Ramadan: ŷ_base + r̂.  Heat-wave τ: passthrough to ŷ_base.',
    accent: true,
  },
];

const reasons = [
  {
    no: '01',
    title: 'A second objective',
    body:
      'The base model was tuned for direct prediction of y with an MSE-adjacent objective. The head is retrained on the L1 residual target with its own hyperparameters — picking up nonlinear interactions the first stage failed to extract.',
  },
  {
    no: '02',
    title: 'Domain features the base never saw',
    body:
      'For zero-shot TSFMs, the head is the only place Hijri, weather, and lag features ever enter the pipeline. The foundation model stays untouched; the head plugs domain knowledge in after the fact.',
  },
  {
    no: '03',
    title: 'Regime-stratified routing',
    body:
      'The head is trained on Normal + Ramadan rows only; Heat-wave τ values pass through to the bare base. Without this routing, a head trained on Normal-regime bias regresses Heat-wave forecasts by 25–32% on the strong TSFMs.',
  },
  {
    no: '04',
    title: 'In-test 3-fold time-block CV',
    body:
      'Folds 2024-01–05, 2024-06–10, 2024-11–2025-03. For each fold the head is trained on the other two and applied to the held-out one; the three held-out predictions are concatenated. No leakage across folds.',
  },
];

export function ResidualMechanismSlide() {
  return (
    <Slide
      eyebrow="The mechanism · Plan 6"
      title={<>A second stage that learns what the base model missed.</>}
      subtitle="How the LightGBM residual head plugs in, what it sees, and why a single small head can rescue every model in the cohort."
    >
      {/* Top: 4-stage horizontal pipeline */}
      <div className="mt-2">
        <div className="type-label-01 uppercase text-g40 mb-3">Pipeline</div>
        <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr_auto_1fr_auto_1fr] items-stretch gap-px bg-g80 border-subtle lg:bg-transparent lg:border-0 lg:gap-0">
          {stages.map((s, i) => (
            <PipelineStep key={s.no} stage={s} index={i} isLast={i === stages.length - 1} />
          ))}
        </div>
      </div>

      {/* Bottom: 2x2 grid of reasons */}
      <div className="mt-6">
        <div className="type-label-01 uppercase text-g40 mb-3">Why it works</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-g80 border-subtle">
          {reasons.map((r, i) => (
            <motion.article
              key={r.no}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: motionTokens.durationModerate02,
                delay: 0.6 + i * 0.08,
                ease: motionTokens.easingEntrance,
              }}
              className="layer-01 p-5"
            >
              <div className="flex items-baseline gap-3 mb-2">
                <span className="type-code-01 text-blue40 tabular">{r.no}</span>
                <span className="type-heading-02 text-g10">{r.title}</span>
              </div>
              <p className="type-body-01 text-g30 leading-relaxed">{r.body}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </Slide>
  );
}

// Sub-components ------------------------------------------------------------

function PipelineStep({
  stage,
  index,
  isLast,
}: {
  stage: PipelineStage;
  index: number;
  isLast: boolean;
}) {
  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{
          duration: motionTokens.durationModerate02,
          delay: 0.1 + index * 0.1,
          ease: motionTokens.easingEntrance,
        }}
        className={`layer-01 p-4 flex flex-col ${
          stage.accent ? 'border-l-2 border-blue60' : ''
        }`}
      >
        <div className="flex items-baseline gap-3 mb-2">
          <span className="type-code-01 text-blue40 tabular">{stage.no}</span>
          <span className="type-code-01 uppercase text-g40">{stage.tag}</span>
        </div>
        <div className={`type-heading-02 mb-2 ${stage.accent ? 'text-blue40' : 'text-g10'}`}>
          {stage.title}
        </div>
        <div className="type-body-01 text-g30 leading-snug">{stage.body}</div>
      </motion.div>
      {!isLast && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationModerate02,
            delay: 0.2 + index * 0.1,
          }}
          className="hidden lg:flex items-center justify-center px-3 text-g60"
        >
          <ArrowRight size={18} strokeWidth={1.5} />
        </motion.div>
      )}
    </>
  );
}
