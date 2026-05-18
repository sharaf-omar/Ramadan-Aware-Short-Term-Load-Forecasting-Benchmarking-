import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { Check } from 'lucide-react';
import { motion as motionTokens } from '../../design/theme';

/**
 * Vertical Stepper — adapted from reactbits' Stepper.
 *
 * Why not the upstream component:
 *  - reactbits' Stepper is an interactive HORIZONTAL wizard with Back/
 *    Continue buttons and paginated content. We want a STATIC vertical
 *    ladder showing constructed stages with no user navigation.
 *  - The visual language (numbered circular indicator + connector line
 *    that fills as it progresses + check icon on the "complete" terminal
 *    state) is preserved here, rotated to vertical, recoloured to Carbon
 *    tokens (blue60 / g80 / g90), and driven by entrance animation
 *    rather than click state.
 */

export interface StepperItem {
  /** A short heading shown beside the indicator (e.g. the model name). */
  label: ReactNode;
  /** Big metric, monospace, shown beneath the label. */
  metric?: ReactNode;
  /** Helper line explaining the construction. */
  description?: ReactNode;
  /**
   * True for the terminal "this is your headline" step. The indicator
   * becomes filled with the primary accent and shows a check.
   */
  highlight?: boolean;
}

interface StepperProps {
  items: StepperItem[];
  /** Delay before the first row enters, in seconds. */
  startDelay?: number;
  /** Gap between rows entering, in seconds. */
  stagger?: number;
  className?: string;
}

export default function Stepper({
  items,
  startDelay = 0.15,
  stagger = motionTokens.cinemaStagger,
  className = '',
}: StepperProps) {
  return (
    <ol className={`relative ${className}`}>
      {items.map((it, i) => {
        const isLast = i === items.length - 1;
        const rowDelay = startDelay + i * stagger;
        return (
          <motion.li
            key={i}
            initial={{ opacity: 0, x: -16 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{
              duration: motionTokens.durationSlow01,
              delay: rowDelay,
              ease: motionTokens.easingExpressive,
            }}
            className="relative grid grid-cols-[40px_minmax(0,1fr)] gap-x-5 py-4"
          >
            {/* Indicator column — circle + connector to next row */}
            <div className="relative flex justify-center">
              {/* Connector line down to next row (skipped on the last) */}
              {!isLast && (
                <div className="absolute top-9 bottom-[-1.25rem] w-px bg-g80">
                  <motion.div
                    initial={{ scaleY: 0 }}
                    animate={{ scaleY: 1 }}
                    transition={{
                      duration: 0.5,
                      delay: rowDelay + 0.25,
                      ease: motionTokens.easingExpressive,
                    }}
                    style={{ transformOrigin: 'top' }}
                    className={
                      it.highlight || items[i + 1]?.highlight
                        ? 'h-full w-full bg-blue60'
                        : 'h-full w-full bg-g70'
                    }
                  />
                </div>
              )}

              {/* Numbered circular indicator */}
              <motion.div
                initial={{ scale: 0.85 }}
                animate={{ scale: 1 }}
                transition={{
                  duration: motionTokens.durationSlow01,
                  delay: rowDelay + 0.05,
                  ease: motionTokens.easingExpressive,
                }}
                className={`relative z-10 flex h-8 w-8 items-center justify-center border ${
                  it.highlight
                    ? 'bg-blue60 border-blue60 text-white'
                    : 'bg-g90 border-g70 text-g30'
                }`}
              >
                {it.highlight ? (
                  <Check size={14} strokeWidth={2.4} />
                ) : (
                  <span className="type-code-01 tabular">
                    {String(i + 1).padStart(2, '0')}
                  </span>
                )}
              </motion.div>
            </div>

            {/* Content column */}
            <div
              className={`pb-4 ${
                it.highlight ? 'border-l-2 border-blue60 -ml-5 pl-5 bg-g90' : ''
              }`}
            >
              <div className="grid grid-cols-12 gap-4 items-baseline">
                <div className="col-span-3 lg:col-span-2">
                  {it.metric != null && (
                    <span
                      className={`type-display-01 tabular leading-none ${
                        it.highlight ? 'text-blue40' : 'text-g10'
                      }`}
                    >
                      {it.metric}
                    </span>
                  )}
                </div>
                <div className="col-span-9 lg:col-span-5">
                  <div className="type-heading-02 text-g10">{it.label}</div>
                  {it.highlight && (
                    <div className="type-code-01 uppercase text-blue40 mt-1">
                      Headline
                    </div>
                  )}
                </div>
                {it.description && (
                  <div className="col-span-12 lg:col-span-5 type-body-01 text-g30">
                    {it.description}
                  </div>
                )}
              </div>
            </div>
          </motion.li>
        );
      })}
    </ol>
  );
}
