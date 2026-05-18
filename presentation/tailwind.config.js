/** @type {import('tailwindcss').Config} */
// IBM Carbon dark theme (G100) tokens.
// Reference: https://carbondesignsystem.com/elements/color/tokens
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Carbon gray scale
        g10:  '#f4f4f4',  // text-primary on dark
        g20:  '#e0e0e0',
        g30:  '#c6c6c6',  // text-secondary
        g40:  '#a8a8a8',  // text-helper
        g50:  '#8d8d8d',
        g60:  '#6f6f6f',  // border-strong
        g70:  '#525252',  // layer-03
        g80:  '#393939',  // layer-02 / border-subtle
        g90:  '#262626',  // layer-01
        g100: '#161616',  // background

        // Carbon interactive / focus
        blue60: '#0f62fe', // primary interactive
        blue50: '#4589ff',
        blue40: '#78a9ff',
        blue70: '#0043ce',

        // Carbon support (used sparingly, only for data semantics)
        red40:    '#ff8389',
        red50:    '#fa4d56',
        green40:  '#42be65',
        yellow30: '#f1c21b',
        purple50: '#a56eff',
        cyan40:   '#33b1ff',
        teal40:   '#08bdba',
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      borderRadius: {
        // Carbon: sharp by default. 0 / 1px / 2px max.
        none: '0',
        sm:   '1px',
        DEFAULT: '0',
      },
      letterSpacing: {
        tight:   '-0.01em',
        carbon:  '0.16px',  // Carbon body type tracking
        eyebrow: '0.32px',  // Carbon label-01 tracking
      },
    },
  },
  corePlugins: {
    // Disable container so we use explicit grid via geometry.
    container: false,
  },
  plugins: [],
};
