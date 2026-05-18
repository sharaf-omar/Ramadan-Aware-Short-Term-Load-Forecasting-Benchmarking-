import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { CountUp } from '../components/CountUp';
import { EmphasisBar } from '../components/EmphasisBar';
import { LeaderboardChart } from '../components/charts/LeaderboardChart';
import { headline } from '../data/results';
import { motion as motionTokens } from '../design/theme';

const notes = [
  {
    tag: 'Top 6',
    body: 'All six leaders are composites — no bare single model cracks the top tier.',
  },
  {
    tag: 'Rank 7',
    body:
      'LightGBM-hijri + residual head (MAE 940.4) is the strongest single entry — but it is LightGBM-on-LightGBM-residuals, a methodological composite.',
  },
  {
    tag: 'CI overlap',
    body:
      'In the strong tier, Chronos (968.9), LightGBM-hijri (979.0), and Time-MoE (985.9) all sit inside each other’s 95% intervals.',
  },
];

export function HeadlineSlide() {
  return (
    <Slide
      eyebrow="Headline"
      title={
        <>
          The meta-router lands at{' '}
          <span className="text-blue40 tabular">
            <CountUp value={headline.metaRouterMAE} decimals={1} />
          </span>{' '}
          MW.
        </>
      }
      subtitle={
        <>
          A{' '}
          <span className="text-blue40">
            −<CountUp value={headline.improvementPct} decimals={1} />%
          </span>{' '}
          improvement over the strongest single bare model (Chronos-Bolt L = 720,
          MAE 968.9), and{' '}
          <span className="text-blue40">−14.3%</span> over the tuned
          LightGBM-hijri tabular incumbent. 95% CI [750.8, 948.7].
        </>
      }
    >
      <div className="grid grid-cols-12 gap-x-8 h-[calc(100%-1rem)] mt-2">
        {/* Live chart */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.2,
          }}
          className="col-span-12 lg:col-span-8 layer-01 border-subtle p-4 min-h-[440px]"
        >
          <LeaderboardChart />
        </motion.div>

        {/* Right column: notes */}
        <div className="col-span-12 lg:col-span-4 flex flex-col">
          <div className="type-label-01 uppercase text-g40 mb-3">Read the chart</div>
          <div className="border-t border-g80 flex-1">
            {notes.map((n, i) => (
              <motion.div
                key={n.tag}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.35 + i * 0.1,
                }}
                className="border-b border-g80 py-4"
              >
                <div className="type-code-01 uppercase text-blue40 mb-1.5">
                  {n.tag}
                </div>
                <div className="type-body-01 text-g30 leading-relaxed">
                  {n.body}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </Slide>
  );
}
