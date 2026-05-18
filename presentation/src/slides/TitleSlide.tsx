import { motion } from 'framer-motion';
import { Slide } from '../components/Slide';
import SplitText from '../components/reactbits/SplitText';
import { motion as motionTokens } from '../design/theme';

export function TitleSlide() {
  return (
    <Slide align="top">
      <div className="flex flex-col h-full justify-between max-w-[1200px]">
        {/* Top: institutional label */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: motionTokens.durationModerate02 }}
          className="type-label-01 uppercase text-blue40"
        >
          Egypt University of Informatics · Capstone 2026
        </motion.div>

        {/* Middle: title block — Split Text gives each character its own
            staggered fade-up, lands at ~700 ms total. */}
        <div className="py-6">
          <SplitText
            text="Beyond Blackouts"
            tag="h1"
            splitType="chars"
            delay={45}
            duration={0.55}
            ease={motionTokens.easingExpressive as any}
            from={{ opacity: 0, y: 40 }}
            to={{ opacity: 1, y: 0 }}
            className="text-g10 block"
            textAlign="left"
            style={{ fontSize: 'clamp(2.5rem, 6vw, 5.5rem)', lineHeight: 1.05, fontWeight: 300 }}
          />

          <motion.p
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              duration: motionTokens.durationSlow01,
              delay: 0.1,
              ease: motionTokens.easingEntrance,
            }}
            className="mt-6 type-body-02 text-g30 max-w-[60ch]"
          >
            When time-series foundation models meet calendar-driven regime
            shifts, MENA-grid load prediction, and geographically tuned
            post-hoc residual correction.
          </motion.p>
        </div>

        {/* Bottom: authors + supervisor — left-aligned table style */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: motionTokens.durationSlow01,
            delay: 0.25,
          }}
          className="border-t border-g80 pt-6 grid grid-cols-1 lg:grid-cols-2 gap-8"
        >
          {/* Authors */}
          <div>
            <div className="type-label-01 uppercase text-g40 mb-4">Authors</div>
            <div className="grid grid-cols-2 gap-x-8 gap-y-2">
              {[
                ['Omar Shafiy', '23-201356'],
                ['Eiad Essam', '23-101108'],
                ['Omar Sharaf', '24-101236'],
                ['Shady Adham', '23-101027'],
              ].map(([name, id]) => (
                <div key={id} className="flex items-baseline justify-between gap-3">
                  <span className="type-body-01 text-g10">{name}</span>
                  <span className="type-code-01 text-g40 tabular">{id}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Supervisor + date */}
          <div className="lg:border-l lg:border-g80 lg:pl-8 grid grid-cols-2 gap-x-8">
            <div>
              <div className="type-label-01 uppercase text-g40 mb-4">Supervisor</div>
              <div className="type-body-01 text-g10">Prof. Mohamed Taher Elrafaie</div>
            </div>
            <div>
              <div className="type-label-01 uppercase text-g40 mb-4">Date</div>
              <div className="type-body-01 text-g10 tabular">May 2026</div>
            </div>
          </div>
        </motion.div>
      </div>
    </Slide>
  );
}
