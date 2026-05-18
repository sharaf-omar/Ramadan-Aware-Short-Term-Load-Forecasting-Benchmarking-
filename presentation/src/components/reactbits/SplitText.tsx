import { ElementType, useRef, useMemo } from 'react';
import { motion, useInView } from 'framer-motion';

/**
 * SplitText — drop-in shape of reactbits' SplitText, but framer-motion only.
 *
 * Why not the upstream component:
 *  - The reactbits/GSAP version relies on GSAP's commercial SplitText plugin
 *    (requires a Club GreenSock license for production) plus ScrollTrigger.
 *  - This file reproduces the same visual effect (char/word stagger from
 *    a "from" state to a "to" state) using framer-motion, which we already
 *    depend on. No new dependencies, no licensing.
 *
 * API matches the reactbits props for the subset we use here.
 */

export interface SplitTextProps {
  text: string;
  className?: string;
  /** Delay between siblings in ms (matches reactbits default of 50). */
  delay?: number;
  /** Duration of each char/word's animation in seconds. */
  duration?: number;
  /** Bezier easing. */
  ease?: [number, number, number, number];
  /** How to slice the source text. */
  splitType?: 'chars' | 'words';
  /** Initial state for each piece. */
  from?: { opacity?: number; y?: number; x?: number };
  /** Resting state for each piece. */
  to?: { opacity?: number; y?: number; x?: number };
  /** Intersection threshold to begin animating, 0–1. */
  threshold?: number;
  /** rootMargin string passed to framer-motion's useInView. */
  rootMargin?: string;
  /** HTML tag rendered as the parent. */
  tag?: 'h1' | 'h2' | 'h3' | 'h4' | 'h5' | 'h6' | 'p' | 'span' | 'div';
  textAlign?: React.CSSProperties['textAlign'];
  /** Inline styles applied to the parent tag (e.g. font-size, line-height). */
  style?: React.CSSProperties;
  onLetterAnimationComplete?: () => void;
}

const SplitText = ({
  text,
  className = '',
  delay = 50,
  duration = 0.7,
  ease = [0.2, 0, 0.38, 0.9],
  splitType = 'chars',
  from = { opacity: 0, y: 24 },
  to = { opacity: 1, y: 0 },
  threshold = 0.1,
  rootMargin = '-50px',
  tag = 'p',
  textAlign,
  style,
  onLetterAnimationComplete,
}: SplitTextProps) => {
  const ref = useRef<HTMLElement>(null);
  // useInView's `margin` is the rootMargin equivalent; `amount` is threshold.
  const inView = useInView(ref, {
    once: true,
    amount: threshold,
    margin: rootMargin as any,
  });

  // Stable split — recompute only when text or splitType changes.
  const pieces = useMemo(() => {
    if (splitType === 'words') {
      // Preserve whitespace between words for natural wrap.
      return text.split(/(\s+)/);
    }
    return Array.from(text);
  }, [text, splitType]);

  const Tag = tag as ElementType;
  const stagger = delay / 1000;
  // Notify on completion of the last piece.
  const total = pieces.filter((p) => p.trim().length > 0).length;
  const lastNonWhitespaceIdx = pieces.reduce(
    (acc, p, i) => (p.trim().length > 0 ? i : acc),
    -1
  );

  return (
    <Tag
      ref={ref as any}
      className={`split-parent inline-block whitespace-normal ${className}`}
      style={{ textAlign, wordWrap: 'break-word', ...style }}
    >
      {pieces.map((piece, i) => {
        // Pure whitespace tokens render as plain text (no animation needed).
        if (piece.trim().length === 0) {
          return <span key={i}>{piece}</span>;
        }
        return (
          <motion.span
            key={i}
            className="split-piece inline-block"
            initial={from as any}
            animate={inView ? (to as any) : (from as any)}
            transition={{
              duration,
              delay: (i / Math.max(total, 1)) * stagger * total,
              ease,
            }}
            onAnimationComplete={
              i === lastNonWhitespaceIdx ? onLetterAnimationComplete : undefined
            }
            style={{ willChange: 'transform, opacity' }}
          >
            {piece}
          </motion.span>
        );
      })}
    </Tag>
  );
};

export default SplitText;
