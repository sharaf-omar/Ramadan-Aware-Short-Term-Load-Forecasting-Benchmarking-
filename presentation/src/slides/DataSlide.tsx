import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import { motion as motionTokens } from '../design/theme';

// Hero stat strip — defensible numbers tied to the dataset.
const stats = [
  { value: '63 195', unit: 'hours',     label: 'Total modelling observations' },
  { value: '81',     unit: 'provinces', label: 'ERA5 weather grid (TR)' },
  { value: '13',     unit: 'features',  label: 'v2 panel columns' },
  { value: '7.25',   unit: 'years',     label: 'Train+val+test span' },
];

const sources = [
  {
    name: 'Turkish national load',
    source: 'EPIAS Transparency Platform',
    span: '2018-01-01 → 2025-03-31',
    note: 'MW-precise hourly, no missing intervals during the modelling window.',
  },
  {
    name: 'ERA5 weather reanalysis',
    source: 'ECMWF Copernicus',
    span: '81 provinces · pop-weighted + southern-cities mean',
    note: '2-m temp, dewpoint, 10-m wind, surface shortwave. Southern mean drives heat-wave detection.',
  },
  {
    name: 'Hijri calendar (Umm al-Qura)',
    source: 'hijridate Python library',
    span: 'is_ramadan · day_of_ramadan · is_eid',
    note: 'Plus four ramadan × hour interactions and two days-to/from-Eid features for the hijri_plusB variant.',
  },
];

// The v2 feature roster — flat list grouped by family.
const featureGroups = [
  { tag: 'Target',    items: ['actual_load'] },
  { tag: 'Calendar',  items: ['hour_sin/cos', 'dow_sin/cos', 'is_weekend'] },
  { tag: 'Weather',   items: ['temp_c', 'dewpoint_c', 'wind_speed', 'solar_rad', 'temp_sq', 'temp_above_35'] },
  { tag: 'Hijri',     items: ['is_ramadan', 'day_of_ramadan', 'is_eid'] },
  { tag: 'Lags',      items: ['y_lag_24h', 'y_lag_168h', 'y_lag_336h'] },
  { tag: 'Rolling',   items: ['y_roll168_mean', 'y_roll168_std'] },
  { tag: 'Interaction', items: ['ramadan_x_heatwave', 'ramadan_x_temp_above_35'] },
];

export function DataSlide() {
  return (
    <Slide
      eyebrow="Data"
      title={<>Why Turkey? It is our closest proxy.</>}
      subtitle="Egypt does not publish hourly load at sufficient resolution. Turkey does — and it shares the same Hijri-driven Ramadan and Eid shifts, the same hot-Mediterranean weather, and an open transparency platform."
    >
      {/* Top: hero stat strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-g80 border-subtle">
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: motionTokens.durationModerate02,
              delay: 0.05 + i * 0.06,
              ease: motionTokens.easingEntrance,
            }}
            className="layer-01 p-5"
          >
            <div className="type-label-01 uppercase text-blue40">{s.label}</div>
            <div className="mt-3 flex items-baseline gap-2">
              <span className="text-g10 tabular type-display-02 leading-none">
                {s.value}
              </span>
              <span className="type-body-01 text-g30">{s.unit}</span>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-6 grid grid-cols-12 gap-x-8">
        {/* Left: data sources */}
        <div className="col-span-12 lg:col-span-7">
          <div className="type-label-01 uppercase text-g40 mb-3">
            Primary data sources
          </div>
          <div className="border-t border-g80">
            {sources.map((s, i) => (
              <motion.div
                key={s.name}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.3 + i * 0.07,
                }}
                className="border-b border-g80 py-4 grid grid-cols-12 gap-4"
              >
                <div className="col-span-12 md:col-span-4">
                  <div className="type-heading-02 text-g10">{s.name}</div>
                  <div className="type-code-01 text-g40 uppercase mt-1">
                    {s.source}
                  </div>
                </div>
                <div className="col-span-12 md:col-span-8">
                  <div className="type-code-01 text-blue40 mb-1.5 tabular">
                    {s.span}
                  </div>
                  <div className="type-body-01 text-g30 leading-relaxed">
                    {s.note}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Chronological split inside the same column */}
          <div className="mt-6">
            <div className="type-label-01 uppercase text-g40 mb-3">
              Chronological split
            </div>
            <div className="layer-01 border-subtle p-4 space-y-3">
              <SplitBar label="Train"      range="2018-01 → 2022-12" pct={62}   hours={43491} />
              <SplitBar label="Validation" range="2023"              pct={12.5} hours={8760} />
              <SplitBar label="Test"       range="2024-01 → 2025-03" pct={15.6} hours={10944} highlight />
            </div>
          </div>
        </div>

        {/* Right: feature roster */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: motionTokens.durationSlow01, delay: 0.45 }}
          className="col-span-12 lg:col-span-5 mt-6 lg:mt-0"
        >
          <div className="type-label-01 uppercase text-g40 mb-3">
            v2 feature panel
          </div>
          <div className="layer-01 border-subtle">
            {featureGroups.map((g, i) => (
              <motion.div
                key={g.tag}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{
                  duration: motionTokens.durationModerate02,
                  delay: 0.55 + i * 0.05,
                }}
                className="grid grid-cols-12 gap-3 px-4 py-2.5 border-b border-g80 last:border-b-0 row-hover"
              >
                <div className="col-span-3 type-code-01 uppercase text-blue40">
                  {g.tag}
                </div>
                <div className="col-span-9 type-code-01 text-g10 flex flex-wrap gap-x-3 gap-y-1">
                  {g.items.map((it) => (
                    <span key={it} className="tabular">{it}</span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>

          {/* Test-window highlights */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: motionTokens.durationSlow01, delay: 0.95 }}
            className="mt-4 grid grid-cols-3 gap-px bg-g80 border-subtle"
          >
            {[
              ['2', 'Ramadans in test'],
              ['2', 'Eid blocks'],
              ['1', 'heat-wave season'],
            ].map(([v, k]) => (
              <div key={k} className="layer-01 p-3 text-center">
                <div className="type-display-01 text-g10 tabular leading-none">{v}</div>
                <div className="type-label-01 uppercase text-g40 mt-1.5">{k}</div>
              </div>
            ))}
          </motion.div>
        </motion.div>
      </div>
    </Slide>
  );
}

function SplitBar({
  label,
  range,
  pct,
  hours,
  highlight,
}: {
  label: string;
  range: string;
  pct: number;
  hours: number;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <div className="flex items-baseline gap-3">
          <span className="type-heading-01 text-g10">{label}</span>
          <span className="type-code-01 text-g40">{range}</span>
        </div>
        <span className="type-code-01 text-g30 tabular">
          {hours.toLocaleString()} h
        </span>
      </div>
      <div className="h-2 bg-g90 relative">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{
            duration: 0.6,
            delay: 0.4,
            ease: [0.2, 0, 0.38, 0.9],
          }}
          className={highlight ? 'h-full bg-blue60' : 'h-full bg-g60'}
        />
      </div>
    </div>
  );
}
