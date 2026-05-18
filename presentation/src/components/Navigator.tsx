import { ChevronLeft, ChevronRight } from 'lucide-react';

interface Props {
  current: number;
  total: number;
  onPrev: () => void;
  onNext: () => void;
  onJump: (i: number) => void;
  title: string;
}

/**
 * Carbon-style bottom navigation bar:
 *   - flat, solid layer-01 surface separated from content by a 1px hairline
 *   - rectangular buttons (no rounding)
 *   - progress indicator is a thin 2px bar, not glowing dots
 *   - interactive color is blue-60
 */
export function Navigator({ current, total, onPrev, onNext, onJump, title }: Props) {
  const progress = ((current + 1) / total) * 100;

  return (
    <>
      {/* Thin top-of-bar progress bar (Carbon ProgressBar style) */}
      <div className="absolute bottom-12 left-0 right-0 z-20 h-[2px] bg-g90">
        <div
          className="h-full bg-blue60 transition-[width] duration-300 ease-[cubic-bezier(0.2,0,0.38,0.9)]"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Bar */}
      <div className="absolute bottom-0 left-0 right-0 z-20 h-12 layer-01 border-bottom-subtle border-t border-t-g80 flex items-center justify-between px-4">
        {/* Left: numeric counter + slide title */}
        <div className="flex items-center gap-4 min-w-0">
          <span className="type-code-01 text-g30 tabular">
            {String(current + 1).padStart(2, '0')} / {String(total).padStart(2, '0')}
          </span>
          <span className="hidden md:block type-body-01 text-g30 truncate max-w-[40ch]">
            {title}
          </span>
        </div>

        {/* Center: jump indicators */}
        <div className="hidden md:flex items-center gap-px">
          {Array.from({ length: total }).map((_, i) => {
            const isCurrent = i === current;
            return (
              <button
                key={i}
                onClick={() => onJump(i)}
                aria-label={`Go to slide ${i + 1}`}
                className={`w-8 h-8 type-code-01 tabular flex items-center justify-center transition-colors duration-100 ${
                  isCurrent
                    ? 'bg-blue60 text-white'
                    : 'text-g40 hover:bg-g80 hover:text-g10'
                }`}
              >
                {String(i + 1).padStart(2, '0')}
              </button>
            );
          })}
        </div>

        {/* Right: prev/next */}
        <div className="flex items-center gap-px">
          <button
            onClick={onPrev}
            disabled={current === 0}
            aria-label="Previous slide"
            className="w-12 h-12 flex items-center justify-center text-g10 hover:bg-g80 disabled:text-g60 disabled:cursor-not-allowed transition-colors duration-100"
          >
            <ChevronLeft size={20} strokeWidth={1.5} />
          </button>
          <button
            onClick={onNext}
            disabled={current === total - 1}
            aria-label="Next slide"
            className="w-12 h-12 flex items-center justify-center text-white bg-blue60 hover:bg-blue70 disabled:bg-g80 disabled:text-g60 disabled:cursor-not-allowed transition-colors duration-100"
          >
            <ChevronRight size={20} strokeWidth={1.5} />
          </button>
        </div>
      </div>
    </>
  );
}
