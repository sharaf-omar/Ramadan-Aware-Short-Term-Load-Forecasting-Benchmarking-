// Carbon-aligned design tokens, used by code that can't read Tailwind
// classes directly (Recharts SVG, inline styles).

export const carbon = {
  background: '#161616', // g100
  layer01:    '#262626', // g90
  layer02:    '#393939', // g80
  layer03:    '#525252', // g70
  borderSubtle: '#393939',
  borderStrong: '#6f6f6f',
  textPrimary:   '#f4f4f4', // g10
  textSecondary: '#c6c6c6', // g30
  textHelper:    '#a8a8a8', // g40
  textPlaceholder: '#6f6f6f', // g60

  // Interactive
  blue60: '#0f62fe',
  blue50: '#4589ff',
  blue40: '#78a9ff',

  // Data / support
  red40:    '#ff8389',
  red50:    '#fa4d56',
  green40:  '#42be65',
  yellow30: '#f1c21b',
  purple50: '#a56eff',
  cyan40:   '#33b1ff',
  teal40:   '#08bdba',
} as const;

// Categorical color order, matches Carbon Charts "default" categorical palette.
// Used when a chart legend exceeds two series.
export const categorical = [
  '#6929c4', // purple-70
  '#1192e8', // cyan-50
  '#005d5d', // teal-70
  '#9f1853', // magenta-70
  '#fa4d56', // red-50
  '#570408', // red-90
  '#198038', // green-60
  '#002d9c', // blue-80
] as const;

// Family → semantic color mapping for the leaderboard / scatter charts.
// Restrained: composites = primary blue (the win), everything else a muted
// support color.
export const familyColor = {
  composite:    '#0f62fe', // blue-60
  tsfm:         '#33b1ff', // cyan-40
  lightgbm:     '#42be65', // green-40
  classical:    '#a8a8a8', // gray-40
  patchtsmixer: '#a56eff', // purple-50
} as const;

// Regime palette — semantic, used in per-regime bars.
export const regimeColor = {
  Normal:      '#78a9ff', // blue-40
  Ramadan:     '#a56eff', // purple-50
  'Heat-wave': '#f1c21b', // yellow-30
  Compound:    '#fa4d56', // red-50
} as const;

// Carbon motion tokens — restrained durations + easings.
// All durations in seconds for framer-motion.
export const motion = {
  durationFast01: 0.07,
  durationFast02: 0.11,
  durationModerate01: 0.15,
  durationModerate02: 0.24,
  durationSlow01: 0.4,
  durationSlow02: 0.7,
  easingStandard:  [0.2, 0, 0.38, 0.9] as const,  // standard productive
  easingEntrance:  [0, 0, 0.38, 0.9] as const,    // entrance productive
  easingExit:      [0.2, 0, 1, 0.9] as const,     // exit productive

  // Cinematic-pacing tokens — used for screen-capture choreography.
  // Lean on Carbon's expressive easing curve (Bezier 0.4, 0.14, 0.3, 1)
  // for a longer, smoother feel without crossing into bouncy / sloppy.
  cinemaSlide: 0.6,        // slide-to-slide
  cinemaHero:  1.0,        // hero number / chart entrance
  cinemaStagger: 0.12,     // gap between staggered children
  easingExpressive: [0.4, 0.14, 0.3, 1] as const,
} as const;
