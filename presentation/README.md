# Beyond Blackouts — Animated Capstone Presentation

A 13-slide React presentation summarising the Ramadan-Aware STLF capstone,
with live Recharts visualizations, framer-motion transitions, and an
aurora-style animated background.

## Quick start

```
cd presentation
npm install
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`).

## Keyboard

| Key | Action |
| --- | --- |
| `→` / `Space` / `PageDown` | Next slide |
| `←` / `PageUp` | Previous slide |
| `Home` / `End` | First / last slide |
| `1`–`9`, `0` | Jump to slide 1–10 |
| `P` | Toggle autoplay (advances each slide on its per-slide dwell) |
| `C` | Toggle clean mode (hides navigator + keyboard hints) |
| `Esc` | Exit autoplay + clean mode |

## Screen-recording a video

The deck has a built-in capture mode that auto-advances slides with a top-of-viewport progress bar.

**Quickest path:**

1. Start the dev or preview server (`npm run dev` or `npm run preview`).
2. Open `http://localhost:5173/?autoplay=1&clean=1` — this lands directly into capture mode (no chrome, autoplay running).
3. Record the browser window with QuickTime, OBS, or your screen-recorder of choice.
4. The deck stops on the closing slide. Total runtime ≈ **2:49**.

**Per-slide dwell times** (configured in `src/App.tsx`):

| # | Slide | Dwell |
| --- | --- | --- |
| 01 | Title | 7 s |
| 02 | Motivation | 14 s |
| 03 | Data | 14 s |
| 04 | Benchmark | 12 s |
| 05 | Headline | 12 s |
| 06 | Per-regime | 11 s |
| 07 | Hijri asymmetry | 12 s |
| 08 | How residual heads work | 14 s |
| 09 | Residual rescue | 14 s |
| 10 | L-sweep | 10 s |
| 11 | Composites | 13 s |
| 12 | Stats | 10 s |
| 13 | Deployment | 14 s |
| 14 | Closing | 10 s |

In clean mode the cursor auto-hides after 2 s of stillness; move the mouse to bring it back. The capture-mode toggles (top-right `Play` / `Clean` buttons) stay around but fade to 30% opacity so they don't appear in the recording — they re-appear on hover if you need them.

## Exporting to PDF

A Playwright-driven exporter renders every slide as a single vector-text PDF page (16:9, 960 × 540 pt) and concatenates them with pdf-lib.

**One-time setup:**

```
npm install                     # if you haven't already
npx playwright install chromium # ~120 MB
```

**Render:**

```
npm run dev                     # start the dev server in one terminal
npm run export:pdf              # in another terminal
```

Output is written to `presentation.pdf` in the project root. Defaults: 14 slides, 2.8 s settle per slide, ~40 s total. Override with CLI flags:

```
npm run export:pdf -- --total=14 --settle=4000 --out=deck.pdf --url=http://localhost:4173
```

Use `--url=http://localhost:4173` with `npm run preview` if you'd rather render against the production build.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--url` | `http://localhost:5173` | Where the dev/preview server is running |
| `--total` | `14` | How many slides to capture |
| `--settle` | `2800` | ms to wait per slide for animations to land |
| `--out` | `presentation.pdf` | Output path |

The exporter uses the deck's `?slide=N&clean=1` URL params to jump directly to each slide and render it in chrome-free mode — no navigator, no keyboard hint, no cursor.

## Build / deploy

```
npm run build      # → dist/
npm run preview    # serve the built bundle locally
```

The `dist/` folder is a static site — host on Vercel, Netlify, GitHub Pages,
or just drop on any static file server.

## Deck structure

1. **Title** — Beyond Blackouts wordmark, authors, supervisor
2. **Motivation** — Egypt blackouts, load-shedding, shop-curfew pilot
3. **Data** — Turkey as proxy, EPIAS + ERA5 + Hijri features, splits
4. **Benchmark** — 31 systems, 4 model families, unified harness
5. **Headline** — Live top-15 forest chart, meta-router-v2 at MAE 838.8
6. **Per-regime** — Live grouped bar chart of top-8 across Normal / Ramadan / Heat-wave
7. **Hijri asymmetry** — Live Ablation A chart; the central finding
8. **Residual rule** — Live scatter of bare MAE vs % improvement
9. **L-sweep** — Live line chart, Ablation C context-length sensitivity
10. **Composites** — Animated step-by-step from 968.9 → 838.8
11. **Stats** — Mini DM heatmap + rigor checklist
12. **Deployment** — Five operating points sorted by cost / accuracy
13. **Closing** — One-sentence summary + repo links

## Stack

- Vite + React 18 + TypeScript
- Tailwind CSS (custom palette in `tailwind.config.js`)
- framer-motion (slide transitions, count-ups, hover lifts)
- Recharts (all live charts)
- lucide-react (icons)
- Google Fonts: Space Grotesk (display), Inter (body), JetBrains Mono (numbers)

All data is hard-coded from the report tables in `src/data/results.ts`, so
the presentation never drifts from the paper. Updating a number is a
one-line edit in that file.
