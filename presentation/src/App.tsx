import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Pause, Play, X } from 'lucide-react';
import { AuroraBg } from './components/AuroraBg';
import { Navigator } from './components/Navigator';
import { Kbd } from './components/Kbd';
import { motion as motionTokens } from './design/theme';

import { TitleSlide } from './slides/TitleSlide';
import { MotivationSlide } from './slides/MotivationSlide';
import { DataSlide } from './slides/DataSlide';
import { BenchmarkSlide } from './slides/BenchmarkSlide';
import { HeadlineSlide } from './slides/HeadlineSlide';
import { PerRegimeSlide } from './slides/PerRegimeSlide';
import { HijriAsymmetrySlide } from './slides/HijriAsymmetrySlide';
import { ResidualMechanismSlide } from './slides/ResidualMechanismSlide';
import { ResidualRuleSlide } from './slides/ResidualRuleSlide';
import { LSweepSlide } from './slides/LSweepSlide';
import { CompositesSlide } from './slides/CompositesSlide';
import { StatsSlide } from './slides/StatsSlide';
import { DeploymentSlide } from './slides/DeploymentSlide';
import { ClosingSlide } from './slides/ClosingSlide';

// Per-slide dwell time in seconds when autoplay is on. Tuned for
// screen-cap pacing: enough time for animations to land and for a
// viewer to scan the content once.
const deck = [
  { id: 'title',      title: 'Beyond Blackouts',           dwell:  7, Component: TitleSlide },
  { id: 'motivation', title: 'The Egyptian problem',       dwell: 14, Component: MotivationSlide },
  { id: 'data',       title: 'Why Turkey — the proxy',     dwell: 14, Component: DataSlide },
  { id: 'benchmark',  title: 'The 31-system benchmark',    dwell: 12, Component: BenchmarkSlide },
  { id: 'headline',   title: 'Headline result',            dwell: 12, Component: HeadlineSlide },
  { id: 'regime',     title: 'Per-regime decomposition',   dwell: 11, Component: PerRegimeSlide },
  { id: 'hijri',      title: 'Hijri-injection asymmetry',  dwell: 12, Component: HijriAsymmetrySlide },
  { id: 'mechanism',  title: 'How residual heads work',    dwell: 14, Component: ResidualMechanismSlide },
  { id: 'residual',   title: 'Residual rescue effect',     dwell: 14, Component: ResidualRuleSlide },
  { id: 'lsweep',     title: 'TSFM context-length sweep',  dwell: 10, Component: LSweepSlide },
  { id: 'composites', title: 'Composite construction',     dwell: 13, Component: CompositesSlide },
  { id: 'stats',      title: 'Statistical rigor',          dwell: 10, Component: StatsSlide },
  { id: 'deploy',     title: 'Deployment recommendations', dwell: 14, Component: DeploymentSlide },
  { id: 'closing',    title: 'Summary and links',          dwell: 10, Component: ClosingSlide },
];

// Read boolean / numeric URL params on first mount so the page can be
// launched directly into capture mode from OBS / QuickTime / Playwright.
function readUrlBool(key: string): boolean {
  const v = new URLSearchParams(window.location.search).get(key);
  return v === '1' || v === 'true';
}

function readUrlInt(key: string): number | null {
  const v = new URLSearchParams(window.location.search).get(key);
  if (v == null) return null;
  const n = parseInt(v, 10);
  return Number.isFinite(n) ? n : null;
}

export default function App() {
  // ?slide=N selects an initial slide (0-indexed). Used by the PDF
  // exporter to jump straight to a single slide per render.
  const initialSlideIndex = (() => {
    const n = readUrlInt('slide');
    if (n == null) return 0;
    return Math.max(0, Math.min(n, 100)); // clamped; deck length checked below
  })();
  const [index, setIndex] = useState(initialSlideIndex);
  const [direction, setDirection] = useState(1);
  const [showHints, setShowHints] = useState(true);

  // Autoplay = timer advances slides automatically.
  // Clean mode = hide navigator chrome (good for screen capture).
  const [autoplay, setAutoplay] = useState(() => readUrlBool('autoplay'));
  const [clean, setClean] = useState(() => readUrlBool('clean') || readUrlBool('autoplay'));

  // Progress within the current slide's dwell window, 0 → 1.
  const [progress, setProgress] = useState(0);
  const startedAtRef = useRef<number>(performance.now());

  const go = useCallback(
    (next: number) => {
      if (next < 0 || next >= deck.length) return;
      setDirection(next > index ? 1 : -1);
      setIndex(next);
    },
    [index]
  );

  const next = useCallback(() => go(index + 1), [go, index]);
  const prev = useCallback(() => go(index - 1), [go, index]);

  // Keyboard nav. `P` toggles autoplay, `C` toggles clean mode, Esc exits both.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {
        e.preventDefault();
        next();
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault();
        prev();
      } else if (e.key === 'Home') {
        e.preventDefault();
        go(0);
      } else if (e.key === 'End') {
        e.preventDefault();
        go(deck.length - 1);
      } else if (e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        setAutoplay((v) => !v);
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        setClean((v) => !v);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        setAutoplay(false);
        setClean(false);
      } else if (/^[0-9]$/.test(e.key)) {
        const target = e.key === '0' ? 9 : parseInt(e.key, 10) - 1;
        if (target < deck.length) go(target);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [next, prev, go]);

  // Autoplay timer: 60Hz progress tick + slide advance when dwell elapses.
  useEffect(() => {
    if (!autoplay) {
      setProgress(0);
      return;
    }
    const dwellMs = deck[index].dwell * 1000;
    startedAtRef.current = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const elapsed = now - startedAtRef.current;
      const t = Math.min(1, elapsed / dwellMs);
      setProgress(t);
      if (t >= 1) {
        if (index < deck.length - 1) {
          go(index + 1);
        } else {
          setAutoplay(false);
        }
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [autoplay, index, go]);

  // Auto-fade keyboard hint
  useEffect(() => {
    const t = setTimeout(() => setShowHints(false), 6500);
    return () => clearTimeout(t);
  }, []);

  // In clean mode, hide the cursor after 2s of stillness for cleaner capture.
  const [cursorHidden, setCursorHidden] = useState(false);
  useEffect(() => {
    if (!clean) {
      setCursorHidden(false);
      return;
    }
    let t = setTimeout(() => setCursorHidden(true), 2000);
    const onMove = () => {
      setCursorHidden(false);
      clearTimeout(t);
      t = setTimeout(() => setCursorHidden(true), 2000);
    };
    window.addEventListener('mousemove', onMove);
    return () => {
      window.removeEventListener('mousemove', onMove);
      clearTimeout(t);
    };
  }, [clean]);

  const Current = deck[index].Component;

  // Cinematic slide variants: a longer fade combined with a subtle
  // horizontal slide and a barely-perceptible scale-in. Stays inside
  // Carbon's expressive easing curve — no bounce, no overshoot.
  const variants = useMemo(
    () => ({
      enter:  (d: number) => ({ opacity: 0, x: 32 * d, scale: 0.985 }),
      center: { opacity: 1, x: 0, scale: 1 },
      exit:   (d: number) => ({ opacity: 0, x: -32 * d, scale: 0.985 }),
    }),
    []
  );

  // Layout: in clean mode we drop the bottom 48px nav reservation so the
  // slide content fills the whole viewport (better for 16:9 capture).
  const wrapperClasses = [
    'relative w-screen h-screen overflow-hidden bg-g100 text-g10',
    clean ? '' : 'pb-12',
    cursorHidden ? 'cursor-none' : '',
  ]
    .join(' ')
    .trim();

  return (
    <div className={wrapperClasses}>
      <AuroraBg />

      <AnimatePresence mode="wait" custom={direction}>
        <motion.div
          key={deck[index].id}
          custom={direction}
          variants={variants}
          initial="enter"
          animate="center"
          exit="exit"
          transition={{
            duration: motionTokens.cinemaSlide,
            ease: motionTokens.easingExpressive,
          }}
          className={`absolute inset-0 ${clean ? '' : 'bottom-12'}`}
        >
          <Current />
        </motion.div>
      </AnimatePresence>

      {/* Top-of-viewport autoplay progress bar — only when autoplay is on */}
      {autoplay && (
        <div className="absolute top-0 left-0 right-0 h-[3px] z-30 bg-g90">
          <div
            className="h-full bg-blue60"
            style={{
              width: `${progress * 100}%`,
              transition: 'width 60ms linear',
            }}
          />
        </div>
      )}

      {/* Navigator — hidden in clean mode */}
      {!clean && (
        <Navigator
          current={index}
          total={deck.length}
          onPrev={prev}
          onNext={next}
          onJump={go}
          title={deck[index].title}
        />
      )}

      {/* Top-right capture controls — always present, semi-faded in clean mode */}
      <div
        className={`absolute top-4 right-4 z-30 flex items-center gap-2 transition-opacity duration-200 ${
          clean ? 'opacity-30 hover:opacity-100' : 'opacity-100'
        }`}
      >
        <button
          onClick={() => setAutoplay((v) => !v)}
          aria-label={autoplay ? 'Pause autoplay' : 'Start autoplay'}
          title={autoplay ? 'Pause (P)' : 'Play (P)'}
          className="h-8 px-3 flex items-center gap-2 layer-01 border-subtle type-code-01 uppercase text-g10 hover:bg-g80 transition-colors"
        >
          {autoplay ? <Pause size={12} /> : <Play size={12} />}
          {autoplay ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={() => setClean((v) => !v)}
          aria-label={clean ? 'Show chrome' : 'Clean mode'}
          title={clean ? 'Show chrome (C / Esc)' : 'Clean mode for screen capture (C)'}
          className="h-8 px-3 flex items-center gap-2 layer-01 border-subtle type-code-01 uppercase text-g10 hover:bg-g80 transition-colors"
        >
          {clean ? <X size={12} /> : null}
          {clean ? 'Clean · on' : 'Clean'}
        </button>
      </div>

      {/* First-load keyboard hint */}
      <AnimatePresence>
        {showHints && !clean && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: motionTokens.durationModerate02 }}
            className="absolute top-16 right-4 z-20 layer-01 border-subtle px-3 py-2 flex flex-col gap-1 type-code-01 text-g30"
          >
            <div className="flex items-center gap-2">
              <span className="uppercase text-blue40">Nav</span>
              <Kbd>←</Kbd>
              <Kbd>→</Kbd>
              <Kbd>Space</Kbd>
              <Kbd>1</Kbd>
              <span>–</span>
              <Kbd>9</Kbd>
            </div>
            <div className="flex items-center gap-2">
              <span className="uppercase text-blue40">Capture</span>
              <Kbd>P</Kbd>
              <span>play</span>
              <Kbd>C</Kbd>
              <span>clean</span>
              <Kbd>Esc</Kbd>
              <span>exit</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
