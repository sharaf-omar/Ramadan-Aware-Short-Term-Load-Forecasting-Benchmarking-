import { motion } from 'framer-motion';
import { ReactNode } from 'react';
import { motion as motionTokens } from '../design/theme';

interface Props {
  label: string;
  children: ReactNode;
  footnote?: string;
  delay?: number;
}

/**
 * Carbon "Tile" / metric card. Flat layer-01 surface, sharp corners,
 * left-aligned label / metric / helper text. No hover scale, no glow.
 */
export function StatCard({ label, children, footnote, delay = 0 }: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: motionTokens.durationModerate02,
        delay,
        ease: motionTokens.easingEntrance,
      }}
      className="layer-01 p-5 border-subtle"
    >
      <div className="type-label-01 uppercase text-g40">{label}</div>
      <div className="mt-3 text-g10">{children}</div>
      {footnote && (
        <div className="mt-2 type-helper-01 text-g40 leading-snug">
          {footnote}
        </div>
      )}
    </motion.div>
  );
}
