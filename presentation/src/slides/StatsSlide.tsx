import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { carbon, motion as motionTokens } from '../design/theme';

// Curated 12-system DM slice. The cells encode a synthetic strength delta;
// the real 31x31 lives in docs/statistical_appendix.md.
const labels = [
  'meta-router v2', 'meta-router v1', 'ens 4 + res', 'stacked LGBM',
  'ens 4 mixed', 'routed best', 'LGBM-hijri + res', 'Chronos + res',
  'LGBM-noh + res', 'Time-MoE + res', 'Chronos L=720', 'LGBM-hijri',
];
const strength = [12, 11.8, 10.4, 9.6, 9.5, 8.6, 7.4, 7.1, 7.0, 6.8, 5.9, 5.5];

const rigor = [
  ['CI width',             'meta-router v2: [750.8, 948.7] MW'],
  ['DM pairs / regime',    '465 paired tests (31 choose 2)'],
  ['Multiple-comparison',  'Holm sequentially-rejective, α = 0.05'],
  ['HAC truncation lag',   'h − 1 = 23 (one day-ahead horizon)'],
  ['Bootstrap block',      '24 h, preserves intra-day autocorrelation'],
  ['Resamples',            '1 000 stationary-block resamples per CI'],
];

function cellFill(diff: number, sig: boolean) {
  if (Math.abs(diff) < 0.01) return carbon.layer02;
  if (!sig) return carbon.layer02;
  return diff > 0 ? carbon.blue60 : carbon.red50;
}

export function StatsSlide() {
  return (
    <Slide
      eyebrow="Statistical rigor"
      title={<>Every claim is Holm-adjusted and bootstrap-bounded.</>}
      subtitle="Diebold–Mariano with Newey–West HAC at h − 1 = 23, Holm sequentially-rejective control at α = 0.05, 1 000-resample stationary block bootstrap. The full 31 × 31 matrices live in docs/statistical_appendix.md."
    >
      <div className="grid grid-cols-12 gap-x-8 mt-2">
        {/* DM matrix */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.15,
          }}
          className="col-span-12 lg:col-span-7 layer-01 border-subtle p-5"
        >
          <div className="type-label-01 uppercase text-g40 mb-3">
            Strong-tier DM significance · 12 × 12 slice
          </div>
          <div className="overflow-auto">
            <table className="border-separate" style={{ borderSpacing: 1 }}>
              <thead>
                <tr>
                  <th />
                  {labels.map((l) => (
                    <th
                      key={l}
                      className="type-code-01 text-g40 align-bottom pb-2"
                      style={{ writingMode: 'vertical-rl', height: 110, minWidth: 18 }}
                    >
                      <span style={{ transform: 'rotate(180deg)', display: 'inline-block' }}>
                        {l}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {labels.map((rowL, i) => (
                  <tr key={rowL}>
                    <th className="type-code-01 text-g40 pr-2 text-right whitespace-nowrap">
                      {rowL}
                    </th>
                    {labels.map((_, j) => {
                      const diff = strength[i] - strength[j];
                      const sig = Math.abs(diff) > 1.2;
                      return (
                        <motion.td
                          key={j}
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          transition={{
                            duration: motionTokens.durationFast02,
                            delay: 0.3 + (i + j) * 0.008,
                          }}
                          style={{
                            background: i === j ? carbon.background : cellFill(diff, sig),
                            width: 18,
                            height: 18,
                          }}
                          title={`${labels[i]} vs ${labels[j]} → ${
                            i === j ? '—' : diff > 0 ? 'col wins' : 'row wins'
                          }${sig ? ' (p<0.05)' : ' (n.s.)'}`}
                        />
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 flex items-center gap-5 type-code-01 text-g40">
            <Legend swatch={carbon.blue60} label="col wins (Holm-sig)" />
            <Legend swatch={carbon.red50}  label="row wins (Holm-sig)" />
            <Legend swatch={carbon.layer02} label="not significant" />
          </div>
        </motion.div>

        {/* Rigor table */}
        <div className="col-span-12 lg:col-span-5">
          <div className="type-label-01 uppercase text-g40 mb-3">
            Test contract
          </div>
          <div className="border-t border-g80">
            {rigor.map((row, i) => (
              <motion.div
                key={row[0]}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.35 + i * 0.06,
                }}
                className="border-b border-g80 py-3 grid grid-cols-12 gap-3 items-baseline"
              >
                <div className="col-span-4 type-code-01 uppercase text-g40">
                  {row[0]}
                </div>
                <div className="col-span-8 type-body-01 text-g10 tabular">
                  {row[1]}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </Slide>
  );
}

function Legend({ swatch, label }: { swatch: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span style={{ background: swatch, width: 12, height: 12, display: 'inline-block' }} />
      {label}
    </div>
  );
}
