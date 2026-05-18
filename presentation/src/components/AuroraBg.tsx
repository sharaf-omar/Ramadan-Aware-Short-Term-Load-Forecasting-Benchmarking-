/**
 * Carbon-style background: solid g100 with a faint regular dot pattern.
 * No gradients, no animated orbs, no blur. The pattern is barely visible
 * and exists only to keep large empty surfaces from feeling lifeless.
 */
export function AuroraBg() {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {/* Solid background — explicit so we don't depend on body class */}
      <div className="absolute inset-0 bg-g100" />
      {/* Sparse 1px dot grid, 24px pitch, ~3% opacity */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.03]"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern id="carbon-dots" width="24" height="24" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="1" fill="#f4f4f4" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#carbon-dots)" />
      </svg>
    </div>
  );
}
