// All numbers taken from the final report (docs/statistical_appendix.md
// and report/main.tex tables). Kept as one source of truth so the
// presentation never drifts from the paper.

export type Family = 'composite' | 'tsfm' | 'lightgbm' | 'classical' | 'patchtsmixer';

export interface SystemRow {
  rank: number;
  name: string;
  shortName: string;
  family: Family;
  mae: number;
  ciLow: number;
  ciHigh: number;
}

// Top-15 leaderboard, aggregate regime.
export const leaderboard: SystemRow[] = [
  { rank: 1,  name: 'meta-router-v2',                shortName: 'meta-router v2',     family: 'composite',    mae: 838.8, ciLow: 750.8, ciHigh: 948.7 },
  { rank: 2,  name: 'meta-router v1',                shortName: 'meta-router v1',     family: 'composite',    mae: 840.9, ciLow: 754.5, ciHigh: 953.0 },
  { rank: 3,  name: 'ensemble-top4-residual',        shortName: 'ensemble (4 + res)', family: 'composite',    mae: 872.4, ciLow: 783.9, ciHigh: 984.9 },
  { rank: 4,  name: 'stacked-lgbm meta-learner',     shortName: 'stacked LGBM',       family: 'composite',    mae: 891.0, ciLow: 793.8, ciHigh: 1011.3 },
  { rank: 5,  name: 'ensemble-top4 (mixed)',         shortName: 'ensemble (4 mixed)', family: 'composite',    mae: 891.4, ciLow: 798.6, ciHigh: 1009.8 },
  { rank: 6,  name: 'routed best-per-regime',        shortName: 'routed best',        family: 'composite',    mae: 916.0, ciLow: 824.2, ciHigh: 1036.9 },
  { rank: 7,  name: 'LightGBM-hijri + residual',     shortName: 'LGBM-hijri + res',   family: 'lightgbm',     mae: 940.4, ciLow: 848.5, ciHigh: 1044.1 },
  { rank: 8,  name: 'Chronos-Bolt L=720 + residual', shortName: 'Chronos + res',      family: 'tsfm',         mae: 948.5, ciLow: 846.9, ciHigh: 1072.4 },
  { rank: 9,  name: 'LightGBM-nohijri + residual',   shortName: 'LGBM-nohijri + res', family: 'lightgbm',     mae: 950.7, ciLow: 855.2, ciHigh: 1063.1 },
  { rank: 10, name: 'Time-MoE L=720 + residual',     shortName: 'Time-MoE + res',     family: 'tsfm',         mae: 954.5, ciLow: 851.7, ciHigh: 1079.0 },
  { rank: 11, name: 'Chronos-Bolt-Base L=720',       shortName: 'Chronos L=720',      family: 'tsfm',         mae: 968.9, ciLow: 868.8, ciHigh: 1097.9 },
  { rank: 12, name: 'LightGBM-hijri (seed 44)',      shortName: 'LGBM-hijri',         family: 'lightgbm',     mae: 979.0, ciLow: 890.2, ciHigh: 1079.9 },
  { rank: 13, name: 'Time-MoE-200M L=720',           shortName: 'Time-MoE L=720',     family: 'tsfm',         mae: 985.9, ciLow: 878.2, ciHigh: 1119.7 },
  { rank: 14, name: 'LightGBM-nohijri (seed 44)',    shortName: 'LGBM-nohijri',       family: 'lightgbm',     mae: 1003.3, ciLow: 907.5, ciHigh: 1113.3 },
  { rank: 15, name: 'TimesFM-2.5 L=168',             shortName: 'TimesFM L=168',      family: 'tsfm',         mae: 1173.2, ciLow: 1057.2, ciHigh: 1313.2 },
];

// Per-regime winners.
export const perRegime = [
  { regime: 'Normal',    composite: 775.1, bestSingle: 't-MoE 880',  composite_name: 'ensemble-top4-residual', single_name: 'LightGBM-hijri' },
  { regime: 'Ramadan',   composite: 799.9, bestSingle: 815.7,        composite_name: 'meta-router-v2',          single_name: 'LightGBM-hijri' },
  { regime: 'Heat-wave', composite: 1206.0, bestSingle: 1208.7,      composite_name: 'meta-router-v2',          single_name: 'Chronos-Bolt L=720' },
];

// Ablation A: Hijri-feature injection delta on Ramadan MAE.
// Negative = Hijri helps Ramadan; positive = Hijri hurts Ramadan.
export interface HijriDelta {
  model: string;
  injection: 'residual head' | 'feature engineering' | 'HF covariate API' | 'cross-channel mix';
  deltaPct: number; // Ramadan-MAE delta in %
}

export const hijriDelta: HijriDelta[] = [
  { model: 'MSTL+ETS',     injection: 'feature engineering', deltaPct: -28.0 },
  { model: 'LightGBM',     injection: 'feature engineering', deltaPct: -7.2 },
  { model: 'Chronos',      injection: 'residual head',       deltaPct: -22.4 },
  { model: 'TimesFM',      injection: 'residual head',       deltaPct: -15.1 },
  { model: 'Moirai',       injection: 'residual head',       deltaPct: -12.8 },
  { model: 'Time-MoE',     injection: 'residual head',       deltaPct: -9.0 },
  { model: 'SARIMAX',      injection: 'feature engineering', deltaPct: -0.2 },
  { model: 'PatchTSMixer', injection: 'cross-channel mix',   deltaPct: 4.0 },
  { model: 'TimesFM',      injection: 'HF covariate API',    deltaPct: 14.0 },
  { model: 'Moirai',       injection: 'HF covariate API',    deltaPct: 25.0 },
];

// Plan 6: Residual correction monotonic rule.
// Bare MAE vs. corrected MAE, sorted by improvement magnitude.
export interface ResidualPoint {
  model: string;
  bareMAE: number;
  correctedMAE: number;
  deltaPct: number;
  family: Family;
}

export const residualImpact: ResidualPoint[] = [
  { model: 'SARIMAX-hijri',         bareMAE: 2485.9, correctedMAE: 1299.3, deltaPct: -47.7, family: 'classical' },
  { model: 'PatchTSMixer L=168',    bareMAE: 1552.7, correctedMAE: 1045.8, deltaPct: -32.6, family: 'patchtsmixer' },
  { model: 'Moirai L=336',          bareMAE: 1727.1, correctedMAE: 1317.2, deltaPct: -23.7, family: 'tsfm' },
  { model: 'MSTL+ETS-hijri',        bareMAE: 1527.5, correctedMAE: 1364.9, deltaPct: -10.6, family: 'classical' },
  { model: 'TimesFM L=168',         bareMAE: 1173.2, correctedMAE: 1057.5, deltaPct:  -9.9, family: 'tsfm' },
  { model: 'LightGBM-nohijri',      bareMAE: 1003.3, correctedMAE:  950.7, deltaPct:  -5.3, family: 'lightgbm' },
  { model: 'LightGBM-hijri',        bareMAE:  979.0, correctedMAE:  940.4, deltaPct:  -4.0, family: 'lightgbm' },
  { model: 'Time-MoE L=720',        bareMAE:  985.9, correctedMAE:  954.5, deltaPct:  -3.2, family: 'tsfm' },
  { model: 'Chronos-Bolt L=720',    bareMAE:  968.9, correctedMAE:  948.5, deltaPct:  -2.1, family: 'tsfm' },
];

// Ablation C: TSFM context-length sweep (best-effort numbers from report;
// L-sweep CSVs in data/predictions/).
export const lSweep = [
  { L: 96,  Chronos: 1048.5, TimesFM: 1224.7, Moirai: 1893.0, TimeMoE: 1098.4 },
  { L: 168, Chronos: 1015.2, TimesFM: 1173.2, Moirai: 1812.5, TimeMoE: 1056.7 },
  { L: 336, Chronos:  992.4, TimesFM: 1375.8, Moirai: 1727.1, TimeMoE: 1004.1 },
  { L: 720, Chronos:  968.9, TimesFM: 1492.0, Moirai: 1808.2, TimeMoE:  985.9 },
];

// Headline summary numbers used in count-ups and big stats.
export const headline = {
  metaRouterMAE: 838.8,
  chronosBareMAE: 968.9,
  lgbmHijriMAE: 979.0,
  improvementPct: 13.4,
  improvementVsLgbmPct: 14.3,
  numSystems: 31,
  testHours: 10944,
  numFigures: 11,
  numTests: 152,
  numPlans: 7,
  windowYears: 8,
  dmPairs: 31 * 30 / 2,
} as const;
