"""
spatial_weights.py
==================
Population-weighted national weather proxy for Turkey.

Collapses ERA5 gridded .nc files into a single hourly time series per variable,
then merges with the EPIAS load CSV to produce the final ML-ready dataset.

ERA5 files expected at:
    data/raw/{variable}_{year}.nc   (e.g. t2m_2018.nc, ssrd_2022.nc)

Load CSV expected at:
    data/raw/electricity_consumption_2018_2025.csv

Output written to:
    data/processed/weather_proxy.csv          (weather only, for inspection)
    data/processed/final_training_set_v1.csv  (merged load + weather)

Usage:
    python spatial_weights.py
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parents[2]   # project root
RAW_DIR       = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LOAD_CSV    = PROCESSED_DIR / "epias_processed_final.csv"   # preprocessed EPIAS output
WEATHER_CSV = PROCESSED_DIR / "weather_proxy.csv"
OUTPUT_CSV  = PROCESSED_DIR / "final_training_set_v1.csv"

# ── ERA5 configuration ─────────────────────────────────────────────────────────
# Internal NetCDF variable names produced by CDS reanalysis downloads.
# Keys = filename prefix == xarray variable name inside the .nc (confirmed from
# your download script which uses 'product_type': 'reanalysis').
ERA5_VAR_MAP: dict[str, str] = {
    "t2m":  "t2m",
    "d2m":  "d2m",
    "u10":  "u10",
    "v10":  "v10",
    "ssrd": "ssrd",
}

# CDS reanalysis ssrd is accumulated within each UTC day and resets at 00:00 UTC.
# diff() turns it into hourly increments (W/m²); negative values at midnight
# reset are physical artefacts and are clipped to 0.
ACCUMULATED_VARS: set[str] = {"ssrd"}

# ── City weights ───────────────────────────────────────────────────────────────
# Five largest electricity-consuming metros in Turkey.
# Weights are normalised programmatically
CITIES = {
    # ── Marmara (37%) ──────────────────────────────────────────────────────
    "Istanbul": {"lat": 41.01, "lon": 28.97, "w": 0.22},  # 18.3% of population, dominant consumer; reduced from pop-share because maritime climate dampens thermal extremes
    "Kocaeli":  {"lat": 40.77, "lon": 29.94, "w": 0.10},  # Turkey's heaviest industry belt (Tüpraş, Petkim, Ford) — 4-5× its pop-share in electricity
    "Bursa":    {"lat": 40.18, "lon": 29.06, "w": 0.05},  # Auto/textile hub; distinct Uludağ microclimate from Istanbul

    # ── Central Anatolia (21%) ──────────────────────────────────────────────
    "Ankara":   {"lat": 39.93, "lon": 32.85, "w": 0.12},  # Capital + 2nd city; continental climate, 38-40°C summers, cold winters
    "Konya":    {"lat": 37.87, "lon": 32.48, "w": 0.05},  # Largest plateau city; hot arid summers critical for heatwave regime
    "Kayseri":  {"lat": 38.72, "lon": 35.49, "w": 0.04},  # Manufacturing at 1,054m; anchors high-altitude heating load signal

    # ── Aegean (7%) ─────────────────────────────────────────────────────────
    "Izmir":    {"lat": 38.42, "lon": 27.14, "w": 0.07},  # 3rd city; Aegean trade/industry, distinct coastal climate

    # ── Mediterranean (13%) ─────────────────────────────────────────────────
    "Adana":    {"lat": 37.00, "lon": 35.32, "w": 0.05},  # Hottest major city (42-44°C); essential for heatwave threshold
    "Antalya":  {"lat": 36.88, "lon": 30.70, "w": 0.04},  # Tourism-driven seasonal demand; Mediterranean coast anchor
    "Mersin":   {"lat": 36.80, "lon": 34.63, "w": 0.04},  # Largest port; humid coastal heat distinct from Adana; Akkuyu nuclear nearby

    # ── Southeast (11%) ─────────────────────────────────────────────────────
    "Gaziantep":{"lat": 37.07, "lon": 37.38, "w": 0.05},  # Fastest-growing industrial city; semi-arid 40°C+ summers
    "Şanlıurfa":{"lat": 37.16, "lon": 38.80, "w": 0.03},  # GAP irrigation pumping load; 43°C+ summers
    "Diyarbakır":{"lat":37.91, "lon": 40.22, "w": 0.03},  # Eastern SE center; covers eastern half of the region

    # ── Black Sea (5%) ──────────────────────────────────────────────────────
    "Samsun":   {"lat": 41.29, "lon": 36.33, "w": 0.03},  # Largest Black Sea city; humid temperate — no other node covers this coast
    "Trabzon":  {"lat": 41.00, "lon": 39.73, "w": 0.02},  # Eastern Black Sea; hydropower-heavy region; mild thermal signal

    # ── Eastern Anatolia (2%) ───────────────────────────────────────────────
    "Erzurum":  {"lat": 39.91, "lon": 41.27, "w": 0.02},  # 1,869m altitude; Turkey's coldest city; anchors extreme-winter heating regime
}

YEARS = range(2018, 2026)   # 2018 – 2025 inclusive

# Canonical hourly UTC index that every output series is aligned to.
FULL_INDEX = pd.date_range(
    start="2018-01-01 00:00",
    end  ="2025-12-31 23:00",
    freq ="h",
    tz   ="UTC",
)


# ──────────────────────────────────────────────────────────────────────────────
def get_weighted_weather(variable: str, years=YEARS) -> pd.Series:
    """
    Build a population-weighted hourly time series for one ERA5 variable.

    Parameters
    ----------
    variable : str
        One of the keys in ERA5_VAR_MAP ('t2m', 'd2m', 'u10', 'v10', 'ssrd').
    years : iterable of int
        Calendar years to process.

    Returns
    -------
    pd.Series
        Hourly, UTC-indexed, named *variable*.
        Years with missing files produce NaN rows (not silent gaps).
    """
    internal_var = ERA5_VAR_MAP[variable]
    total_w      = sum(c["w"] for c in CITIES.values())   # normalisation factor
    yearly: list[pd.Series] = []

    for year in years:
        fpath = RAW_DIR / f"{variable}_{year}.nc"
        if not fpath.exists():
            print(f"  [WARN] Not found — skipping: {fpath.name}")
            continue

        print(f"  {variable} {year} ...", end=" ", flush=True)

        # chunks={"time": -1} keeps one full year in a single dask chunk —
        # fast enough for ~40 MB files and avoids per-step dask overhead.
        ds = xr.open_dataset(fpath, chunks={"time": -1})

        weighted: pd.Series | None = None

        for city_name, city in CITIES.items():
            # Nearest-neighbour lookup on the ERA5 0.25° grid.
            point: pd.Series = (
                ds[internal_var]
                .sel(latitude=city["lat"], longitude=city["lon"], method="nearest")
                .compute()
                .to_series()
            )

            # De-accumulate ssrd: daily accumulation resets at 00:00 UTC.
            # diff() → hourly increment; clip removes the negative reset spike.
            # The very first row becomes NaN after diff() — fill with 0 (pre-dawn).
            if variable in ACCUMULATED_VARS:
                point = point.diff().clip(lower=0).fillna(0)

            w = city["w"] / total_w
            weighted = point * w if weighted is None else weighted + point * w

        yearly.append(weighted)
        ds.close()
        print("done")

    if not yearly:
        raise FileNotFoundError(
            f"No files found for '{variable}' in {RAW_DIR}. "
            "Verify RAW_DIR is pointing at the right folder."
        )

    series = pd.concat(yearly)

    # Ensure UTC-aware index (ERA5 times are UTC but xarray may omit tzinfo).
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    series = series.sort_index().reindex(FULL_INDEX)
    series.name = variable
    return series


# ──────────────────────────────────────────────────────────────────────────────
# Hot southern cities — used for heatwave regime detection.
# Unweighted mean across these 7 captures AC-load-relevant heat stress;
# absolute 35°C threshold fires several times per summer here, unlike the
# pop-weighted national average which is dampened by cooler northern cities.
SOUTHERN_HEAT_CITIES = [
    "Adana", "Şanlıurfa", "Gaziantep", "Diyarbakır",
    "Mersin", "Konya", "Antalya",
]


def build_southern_temp_series(years=YEARS) -> pd.Series:
    """Unweighted-mean t2m series across SOUTHERN_HEAT_CITIES, in °C.

    Returns
    -------
    Hourly UTC-indexed Series named 'temp_c_south'.
    """
    missing = [c for c in SOUTHERN_HEAT_CITIES if c not in CITIES]
    if missing:
        raise KeyError(f"SOUTHERN_HEAT_CITIES missing from CITIES dict: {missing}")

    yearly: list[pd.Series] = []
    for year in years:
        fpath = RAW_DIR / f"t2m_{year}.nc"
        if not fpath.exists():
            print(f"  [WARN] t2m {year} not found - skipping")
            continue
        print(f"  t2m_south {year} ...", end=" ", flush=True)
        ds = xr.open_dataset(fpath)
        per_city = []
        for city_name in SOUTHERN_HEAT_CITIES:
            city = CITIES[city_name]
            pt = (
                ds["t2m"]
                .sel(latitude=city["lat"], longitude=city["lon"], method="nearest")
                .to_series()
            )
            per_city.append(pt)
        mean_kelvin = sum(per_city) / len(per_city)
        yearly.append(mean_kelvin)
        ds.close()
        print("done")

    series = pd.concat(yearly) - 273.15  # K -> °C
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    series = series.sort_index().reindex(FULL_INDEX)
    series.name = "temp_c_south"
    return series


# ──────────────────────────────────────────────────────────────────────────────
def build_weather_proxy(years=YEARS) -> pd.DataFrame:
    """
    Process all five ERA5 variables and return a clean weather DataFrame.

    Output columns
    --------------
    temp_c      2m temperature (°C)
    dewpoint_c  2m dewpoint temperature (°C)
    wind_speed  10m wind speed magnitude (m/s)
    solar_rad   Surface solar radiation downwards (W/m², de-accumulated)
    """
    print("\n=== Building weather proxy ===\n")

    print("[1/4] Temperature (t2m)")
    t2m  = get_weighted_weather("t2m",  years)

    print("\n[2/4] Dewpoint (d2m)")
    d2m  = get_weighted_weather("d2m",  years)

    print("\n[3/4] Wind components (u10 + v10)")
    u10  = get_weighted_weather("u10",  years)
    v10  = get_weighted_weather("v10",  years)

    print("\n[4/4] Solar radiation (ssrd)")
    ssrd = get_weighted_weather("ssrd", years)

    weather = pd.DataFrame(
        {
            "temp_c":     t2m  - 273.15,              # K → °C
            "dewpoint_c": d2m  - 273.15,              # K → °C
            "wind_speed": np.sqrt(u10**2 + v10**2),   # m/s magnitude
            "solar_rad":  ssrd,                        # W/m²
        },
        index=FULL_INDEX,
    )
    weather.index.name = "timestamp"

    # ── Sanity checks ──────────────────────────────────────────────────────
    print("\n── Weather proxy sanity checks ──")
    print(weather.describe().round(2))

    nan_counts = weather.isna().sum()
    if nan_counts.any():
        print(f"\n[WARN] NaN counts per column:\n{nan_counts[nan_counts > 0]}")
    else:
        print("\n[OK] No NaN values in weather proxy")

    assert (weather["temp_c"]    > -60).all() and (weather["temp_c"] < 60).all(), \
        "temp_c out of physical range — check K→°C conversion"
    assert (weather["solar_rad"].dropna() >= 0).all(), \
        "Negative solar_rad — de-accumulation error"
    assert (weather["wind_speed"] >= 0).all(), \
        "Negative wind_speed — impossible"

    print("[OK] Physical range assertions passed")
    return weather


# ──────────────────────────────────────────────────────────────────────────────
def load_epias(path: Path = LOAD_CSV) -> pd.DataFrame:
    """
    Load the preprocessed EPIAS feature file (epias_processed_final.csv).

    Expected schema (produced by the EPIAS preprocessing script):
        timestamp       : UTC-aware ISO timestamp (index after loading)
        actual_load     : MW consumption (already interpolated)
        is_ramadan      : 0/1 Hijri Ramadan indicator
        is_eid          : 0/1 Eid al-Fitr indicator
        day_of_ramadan  : 1–30 within Ramadan, 0 otherwise
        hour            : 0–23
        day_of_week     : 0 (Mon) – 6 (Sun)
        month           : 1–12
        load_lag_24h    : lag-24 h actual load
        load_lag_168h   : lag-168 h actual load
        load_rolling_*  : rolling mean / std features

    Returns
    -------
    pd.DataFrame with a UTC-aware DatetimeIndex named 'timestamp'.
    """
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            f"'timestamp' column not found in {path.name}. "
            f"Columns present: {df.columns.tolist()}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    print(f"\n[OK] EPIAS loaded: {len(df):,} rows | columns: {df.columns.tolist()}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def merge_and_save(load_df: pd.DataFrame,
                   weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join weather onto load, interpolate isolated NaN hours, assert
    cleanliness, then write both outputs to disk.
    """
    final = load_df.merge(
        weather_df, left_index=True, right_index=True, how="left"
    )

    weather_cols = ["temp_c", "dewpoint_c", "wind_speed", "solar_rad"]

    # Linear interpolation over time for small isolated gaps only.
    # limit=3 ensures genuinely missing stretches stay as NaN rather than
    # being silently filled with stale values.
    final[weather_cols] = final[weather_cols].interpolate(method="time", limit=3)

    missing = final[weather_cols].isna().sum()
    if missing.any():
        print(f"\n[WARN] NaN rows remaining after interpolation "
              f"(need manual review before training):\n{missing[missing > 0]}")
    else:
        print("[OK] No NaN values in final merged dataset")

    # Save weather proxy separately for EDA / debugging
    weather_df.reset_index().to_csv(WEATHER_CSV, index=False)
    print(f"[OK] Weather proxy  →  {WEATHER_CSV}")

    final.reset_index().to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Final dataset  →  {OUTPUT_CSV}")
    print(f"     Shape: {final.shape} | Columns: {final.columns.tolist()}")

    return final


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    weather_df = build_weather_proxy(years=YEARS)
    load_df    = load_epias(LOAD_CSV)
    final_df   = merge_and_save(load_df, weather_df)

    print("\n=== Preview (first 5 rows) ===")
    print(final_df.head().to_string())
    print("\n=== Done ===")