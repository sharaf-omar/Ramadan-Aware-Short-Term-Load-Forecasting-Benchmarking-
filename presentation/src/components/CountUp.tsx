import { useEffect, useRef, useState } from 'react';
import { motion, useInView } from 'framer-motion';
import { motion as motionTokens } from '../design/theme';

interface Props {
  value: number;
  decimals?: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
}

/**
 * Smoothly counts 0 → value when the element first enters view.
 * Uses requestAnimationFrame for precise final-value formatting.
 * Stays inside Carbon's motion budget (default ~700ms slow-02).
 */
export function CountUp({
  value,
  decimals = 0,
  duration = motionTokens.cinemaHero,
  prefix = '',
  suffix = '',
  className = '',
}: Props) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / (duration * 1000));
      // ease-out quint — slow at the end so the final digits land cleanly.
      const eased = 1 - Math.pow(1 - t, 5);
      setDisplay(value * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, value, duration]);

  const formatted = display.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <motion.span
      ref={ref}
      className={`tabular ${className}`}
      initial={{ opacity: 0 }}
      animate={{ opacity: inView ? 1 : 0 }}
      transition={{ duration: motionTokens.durationModerate01 }}
    >
      {prefix}
      {formatted}
      {suffix}
    </motion.span>
  );
}
