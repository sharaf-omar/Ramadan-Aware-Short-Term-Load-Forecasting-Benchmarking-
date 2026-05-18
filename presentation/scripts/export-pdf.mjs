// Export the slide deck to a single PDF.
//
// Usage:  npm run export:pdf
//
// Workflow:
//   1. Reads --total / env / default for slide count (currently 14).
//   2. Launches headless Chromium via Playwright.
//   3. For each slide index 0..N-1, navigates to ?slide=N&clean=1, waits
//      for animations to settle, then captures the viewport as a one-page
//      PDF buffer.
//   4. Concatenates all per-slide PDFs with pdf-lib and writes
//      presentation.pdf in the cwd.
//
// Requires the dev or preview server to be running on the URL passed
// via --url (default http://localhost:5173).

import { chromium } from 'playwright';
import { PDFDocument } from 'pdf-lib';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

// --- CLI ---------------------------------------------------------------
const args = Object.fromEntries(
  process.argv.slice(2).map((a) => {
    const [k, v] = a.replace(/^--/, '').split('=');
    return [k, v ?? true];
  })
);

const BASE_URL   = args.url   ?? 'http://localhost:5173';
const TOTAL      = Number(args.total ?? 14);
const OUTPUT     = resolve(args.out ?? 'presentation.pdf');
const SETTLE_MS  = Number(args.settle ?? 2800);   // ms to wait per slide
// 16:9 widescreen Slides page size — 1920x1080 native, expressed in
// inches so PDF readers render at vector resolution.
const PAGE_WIDTH  = '13.333in';
const PAGE_HEIGHT = '7.5in';

// --- Run ---------------------------------------------------------------
async function main() {
  console.log(`Exporting ${TOTAL} slides from ${BASE_URL} → ${OUTPUT}`);

  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 2, // sharper raster fallback for any rasterised content
  });
  const page = await ctx.newPage();
  // Keep screen styles instead of switching to print stylesheet.
  await page.emulateMedia({ media: 'screen' });

  const merged = await PDFDocument.create();
  merged.setTitle('Beyond Blackouts — Capstone Presentation');
  merged.setAuthor('Omar Shafiy · Eiad Essam · Omar Sharaf · Shady Adham');

  for (let i = 0; i < TOTAL; i++) {
    const url = `${BASE_URL}/?slide=${i}&clean=1`;
    process.stdout.write(`  [${String(i + 1).padStart(2, '0')}/${TOTAL}] ${url}`);

    await page.goto(url, { waitUntil: 'networkidle' });
    // Give framer-motion + recharts entry animations time to settle.
    await page.waitForTimeout(SETTLE_MS);

    const pdfBuf = await page.pdf({
      printBackground: true,
      width: PAGE_WIDTH,
      height: PAGE_HEIGHT,
      margin: { top: 0, right: 0, bottom: 0, left: 0 },
      preferCSSPageSize: false,
    });

    const doc = await PDFDocument.load(pdfBuf);
    const [pg] = await merged.copyPages(doc, [0]);
    merged.addPage(pg);
    process.stdout.write('  ✓\n');
  }

  await browser.close();
  const bytes = await merged.save();
  await writeFile(OUTPUT, bytes);
  console.log(`\nWrote ${OUTPUT} — ${(bytes.length / 1024).toFixed(1)} KB, ${TOTAL} pages.`);
}

main().catch((err) => {
  console.error('Export failed:', err);
  process.exitCode = 1;
});
