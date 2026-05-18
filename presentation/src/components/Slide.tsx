import { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { motion as motionTokens } from '../design/theme';

interface Props {
  eyebrow?: string;
  title?: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  align?: 'center' | 'top';
}

/**
 * Carbon-style slide: 16-column grid of content within a max width,
 * generous left/right margin, restrained header hierarchy.
 *
 * Typography uses Carbon's productive type scale:
 *   eyebrow  → label-01 (12px, 0.32px tracking, uppercase)
 *   title    → heading-06 (Plex Light 42px) for slides
 *   subtitle → body-02 (16px)
 */
export function Slide({
  eyebrow,
  title,
  subtitle,
  children,
  align = 'top',
}: Props) {
  return (
    <div
      className={`relative w-full h-full px-10 lg:px-16 py-10 flex flex-col ${
        align === 'center' ? 'justify-center' : 'justify-start'
      }`}
    >
      <div className="max-w-[1440px] w-full mx-auto flex flex-col h-full">
        {(eyebrow || title || subtitle) && (
          <header className="mb-8">
            {eyebrow && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: motionTokens.durationModerate01 }}
                className="type-label-01 uppercase text-blue40 mb-4"
              >
                {eyebrow}
              </motion.div>
            )}
            {title && (
              <motion.h1
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: motionTokens.durationSlow01,
                  ease: motionTokens.easingEntrance,
                }}
                className="type-heading-06 lg:type-heading-07 text-g10"
              >
                {title}
              </motion.h1>
            )}
            {subtitle && (
              <motion.p
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{
                  duration: motionTokens.durationSlow01,
                  delay: 0.06,
                  ease: motionTokens.easingEntrance,
                }}
                className="mt-4 type-body-02 text-g30 max-w-[80ch]"
              >
                {subtitle}
              </motion.p>
            )}
          </header>
        )}
        <div className="flex-1 min-h-0">{children}</div>
      </div>
    </div>
  );
}
