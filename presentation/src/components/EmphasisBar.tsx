import { motion } from 'framer-motion';
import { motion as motionTokens } from '../design/theme';

interface Props {
  /** Bar height in pixels. Default 2 (Carbon's hairline emphasis weight). */
  height?: number;
  /** Tailwind background-color class. Default `bg-blue60`. */
  color?: string;
  /** Bar width as a percentage of the parent (0–100). Default 100. */
  widthPct?: number;
  /** Animation delay in seconds. */
  delay?: number;
  /** Animation duration in seconds. Default cinemaHero (1.0 s). */
  duration?: number;
  /** Origin of the wipe. Default 'left'. */
  origin?: 'left' | 'center' | 'right';
}

/**
 * A thin colored bar that draws in from a chosen origin.
 * Used as a Carbon-style emphasis cue beneath hero numbers — no glow,
 * no shadow, just a sliced data accent that lands after the value.
 */
export function EmphasisBar({
  height = 2,
  color = 'bg-blue60',
  widthPct = 100,
  delay = 0,
  duration = motionTokens.cinemaHero,
  origin = 'left',
}: Props) {
  const transformOrigin =
    origin === 'center' ? 'center' : origin === 'right' ? 'right' : 'left';
  return (
    <div className="overflow-hidden" style={{ height }}>
      <motion.div
        initial={{ scaleX: 0 }}
        animate={{ scaleX: widthPct / 100 }}
        transition={{
          duration,
          delay,
          ease: motionTokens.easingExpressive,
        }}
        style={{ transformOrigin, height: '100%', width: '100%' }}
        className={color}
      />
    </div>
  );
}
