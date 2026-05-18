import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { CountUp } from '../components/CountUp';
import { motion as motionTokens } from '../design/theme';

const stats = [
  { value: '1–3',   unit: 'h / day',  label: 'Rolling cuts per feeder' },
  { value: '2',     unit: 'summers',  label: 'Under crisis (2023, 2024)' },
  { value: '~14',   unit: 'GW',       label: 'Peak-hour gap on hottest days' },
  { value: '100M+', unit: 'people',   label: 'Affected across 27 governorates' },
  { value: '2',     unit: 'measures', label: 'Demand-side levers piloted' },
  { value: '≥ 35',  unit: '°C',       label: 'Heat-wave trigger threshold' },
];

const problems = [
  {
    no: '01',
    title: 'Rolling load-shedding',
    body:
      'Egypt’s nationwide takhfīf al-aḥmāl schedule cuts 1–3 hours per feeder per day to contain peak demand against a gas-supply shortfall.',
  },
  {
    no: '02',
    title: 'Unscheduled blackouts',
    body:
      'Reported widely during the July–September peaks of 2023 and 2024 — heat-wave demand cascading past the published schedule.',
  },
  {
    no: '03',
    title: 'Shop & restaurant curfew',
    body:
      'The Cabinet has piloted early-closing rules targeting the post-iftar and late-evening load window — a blunt demand-side instrument.',
  },
];

export function MotivationSlide() {
  return (
    <Slide
      eyebrow="The problem"
      title={<>Two summers under load-shedding.</>}
      subtitle="Why this work is motivated by Egypt — not by the Turkish data we trained on."
    >
      {/* 3 problem cards + the "lever" / payoff column */}
      <div className="grid grid-cols-12 gap-x-8">
        <div className="col-span-12 lg:col-span-8 flex flex-col gap-6">
          {/* Problem cards */}
          <div>
            <div className="type-label-01 uppercase text-g40 mb-3">
              What's already in place
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-px bg-g80 border-subtle">
              {problems.map((p, i) => (
                <motion.article
                  key={p.no}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: motionTokens.durationModerate02,
                    delay: 0.1 + i * 0.07,
                    ease: motionTokens.easingEntrance,
                  }}
                  className="layer-01 p-5 min-h-[200px] flex flex-col"
                >
                  <div className="type-code-01 text-blue40 mb-3 tabular">
                    {p.no}
                  </div>
                  <div className="type-heading-03 text-g10 mb-3">{p.title}</div>
                  <p className="type-body-01 text-g30 leading-relaxed">{p.body}</p>
                </motion.article>
              ))}
            </div>
          </div>

          {/* Stat strip — sits under the problem cards in the same 8-col column */}
          <div>
            <div className="type-label-01 uppercase text-g40 mb-3">
              By the numbers
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-px bg-g80 border-subtle">
              {stats.map((s, i) => (
                <motion.div
                  key={s.label}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{
                    duration: motionTokens.durationModerate02,
                    delay: 0.35 + i * 0.05,
                    ease: motionTokens.easingEntrance,
                  }}
                  className="layer-01 px-4 py-3"
                >
                  <div className="flex items-baseline gap-2">
                    <span className="text-g10 tabular type-display-01 leading-none">
                      {s.value}
                    </span>
                    <span className="type-code-01 text-g30">{s.unit}</span>
                  </div>
                  <div className="mt-2 type-label-01 uppercase text-g40 leading-tight">
                    {s.label}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: the lever — what better forecasts buy you */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.55,
          }}
          className="col-span-12 lg:col-span-4 mt-6 lg:mt-0"
        >
          <div className="type-label-01 uppercase text-g40 mb-3">
            Our lever
          </div>
          <div className="layer-01 border-subtle p-5 h-full flex flex-col">
            <div className="type-code-01 text-blue40 mb-3 tabular">→</div>
            <div className="type-heading-03 text-g10 mb-3">
              Tighter day-ahead forecasts
            </div>
            <p className="type-body-01 text-g30 leading-relaxed">
              A tighter day-ahead forecast is the precondition for replacing
              rolling blackouts and commercial curfews with finer-grained,
              market-based balancing.
            </p>

            <div className="mt-4 pt-4 border-t border-g80">
              <div className="type-label-01 uppercase text-g40 mb-2">
                What it unlocks
              </div>
              <p className="type-body-01 text-g30 leading-relaxed">
                Lower MAE shrinks the spinning-reserve buffer the operator
                must hold, lets CCGT dispatch line up against actual evening
                ramps instead of worst-case envelopes, and turns the
                imported-fuel order book into a planning problem rather than
                a panic procurement.
              </p>
            </div>

            <div className="mt-4 pt-4 border-t border-g80">
              <div className="type-label-01 uppercase text-g40 mb-2">
                Why this work transfers
              </div>
              <p className="type-body-01 text-g30 leading-relaxed">
                Hijri-aware features, regime-stratified residual heads, and
                the meta-router pipeline are all model-agnostic. The moment
                hourly Egyptian load data is published at the same resolution
                EPIAS offers, the entire pipeline ports over with no
                architectural changes.
              </p>
            </div>

            <div className="mt-auto pt-4 border-t border-g80">
              <div className="type-label-01 uppercase text-g40">
                Direction of improvement
              </div>
              <div className="mt-2 flex items-baseline gap-3">
                <span className="type-display-01 text-blue40 tabular leading-none">
                  −13.4%
                </span>
                <span className="type-body-01 text-g30">
                  MAE achievable today
                </span>
              </div>
              <div className="mt-1 type-helper-01 text-g40">
                meta-router-v2 vs strongest single bare model
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </Slide>
  );
}
