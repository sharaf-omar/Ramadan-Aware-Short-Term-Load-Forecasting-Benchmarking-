import { motion } from 'framer-motion';
import { Github, FileText, BookOpen } from 'lucide-react';
import { Slide } from '../components/Slide';
import { CountUp } from '../components/CountUp';
import { EmphasisBar } from '../components/EmphasisBar';
import { headline } from '../data/results';
import { motion as motionTokens } from '../design/theme';

// The two weakest classical baselines in the benchmark, used as the
// composite's "wow" comparison band on the closing slide.
const SARIMAX_HIJRI_MAE = 2485.9;
const MSTL_ETS_HIJRI_MAE = 1527.5;

const compositeVsSarimax =
  ((SARIMAX_HIJRI_MAE - headline.metaRouterMAE) / SARIMAX_HIJRI_MAE) * 100;
const compositeVsMstl =
  ((MSTL_ETS_HIJRI_MAE - headline.metaRouterMAE) / MSTL_ETS_HIJRI_MAE) * 100;

export function ClosingSlide() {
  return (
    <Slide align="top">
      <div className="flex flex-col h-full max-w-[1200px]">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: motionTokens.durationModerate02 }}
          className="type-label-01 uppercase text-blue40"
        >
          Summary
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{
            duration: motionTokens.durationSlow02,
            ease: motionTokens.easingEntrance,
          }}
          className="mt-4 text-g10 max-w-[44ch]"
          style={{ fontSize: 'clamp(1.75rem, 3.2vw, 2.75rem)', lineHeight: 1.18, fontWeight: 300 }}
        >
          A composite of {headline.numSystems} benchmarked systems —
          tied together by post-hoc residual correction and regime-aware
          routing — beats every single bare model.
        </motion.h1>

        {/* Twin headline percentages — the % is the visual lead */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.2,
          }}
          className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-px bg-g80 border-subtle"
        >
          <div className="layer-01 p-7 flex flex-col">
            <div className="type-label-01 uppercase text-blue40">
              Composite vs. baselines
            </div>

            {/* Row 1: SARIMAX-hijri */}
            <div className="mt-5">
              <div
                className="text-blue40 tabular leading-none"
                style={{ fontSize: 'clamp(2.5rem, 5.5vw, 4.25rem)', fontWeight: 300 }}
              >
                −<CountUp value={compositeVsSarimax} decimals={1} duration={1.4} />%
              </div>
              <div className="mt-2 type-body-01 text-g30">
                vs SARIMAX-hijri{' '}
                <span className="text-g10 tabular">({SARIMAX_HIJRI_MAE.toFixed(1)} MAE)</span>
              </div>
              <div className="mt-3">
                <EmphasisBar
                  widthPct={compositeVsSarimax}
                  delay={0.7 + 1.2}
                  duration={0.9}
                />
              </div>
            </div>

            {/* Row 2: MSTL+ETS-hijri */}
            <div className="mt-5 pt-5 border-t border-g80">
              <div
                className="text-blue40 tabular leading-none"
                style={{ fontSize: 'clamp(2.5rem, 5.5vw, 4.25rem)', fontWeight: 300 }}
              >
                −<CountUp value={compositeVsMstl} decimals={1} duration={1.4} />%
              </div>
              <div className="mt-2 type-body-01 text-g30">
                vs MSTL+ETS-hijri{' '}
                <span className="text-g10 tabular">({MSTL_ETS_HIJRI_MAE.toFixed(1)} MAE)</span>
              </div>
              <div className="mt-3">
                <EmphasisBar
                  widthPct={compositeVsMstl}
                  delay={0.95 + 1.2}
                  duration={0.9}
                />
              </div>
            </div>

            <div className="mt-auto pt-4 type-helper-01 text-g40">
              meta-router-v2 at{' '}
              <span className="text-g10 tabular">
                {headline.metaRouterMAE.toFixed(1)} MAE
              </span>
            </div>
          </div>
          <div className="layer-01 p-7">
            <div className="type-label-01 uppercase text-blue40">
              Biggest single-model rescue
            </div>
            <div
              className="mt-4 text-blue40 tabular leading-none"
              style={{ fontSize: 'clamp(3.5rem, 8vw, 6.5rem)', fontWeight: 300 }}
            >
              −<CountUp value={47.7} decimals={1} duration={1.4} />%
            </div>
            <div className="mt-4 type-body-01 text-g30">
              SARIMAX-hijri rescued by one LightGBM head{' '}
              <span className="text-g10 tabular">(2 485.9 → 1 299.3 MAE)</span>
            </div>
            <div className="mt-4">
              <EmphasisBar widthPct={47.7 * 2} delay={1.6} duration={0.9} />
            </div>
          </div>
        </motion.div>

        {/* Bottom row: thank-you on the left, icon resources on the right */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.4,
          }}
          className="mt-auto pt-8 flex items-end justify-between gap-8"
        >
          <div className="type-label-01 uppercase text-g40">
            Thank you · Questions?
          </div>

          <div className="flex flex-col gap-2 items-end">
            <div className="type-label-01 uppercase text-g40 mb-1">Resources</div>
            <ResourceRow
              Icon={Github}
              label="Code repository"
              target="github.com/omarshafiy/Ramadan-Aware-STLF"
            />
            <ResourceRow
              Icon={FileText}
              label="Compiled report PDF"
              target="report/main.pdf"
            />
            <ResourceRow
              Icon={BookOpen}
              label="Statistical appendix"
              target="docs/statistical_appendix.md"
            />
          </div>
        </motion.div>
      </div>
    </Slide>
  );
}

function ResourceRow({
  Icon,
  label,
  target,
}: {
  Icon: typeof Github;
  label: string;
  target: string;
}) {
  return (
    <div className="flex items-center gap-3 row-hover py-1.5 px-2 -mx-2">
      <Icon size={16} strokeWidth={1.5} className="text-g30 shrink-0" />
      <span className="type-body-01 text-g10">{label}</span>
      <span className="type-code-01 text-g40">{target}</span>
    </div>
  );
}
