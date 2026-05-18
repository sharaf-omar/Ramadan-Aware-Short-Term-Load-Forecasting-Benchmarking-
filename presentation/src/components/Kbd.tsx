import { ReactNode } from 'react';

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex items-center justify-center min-w-[1.75em] h-6 px-1.5 type-code-01 layer-02 text-g10 border border-g70">
      {children}
    </kbd>
  );
}
