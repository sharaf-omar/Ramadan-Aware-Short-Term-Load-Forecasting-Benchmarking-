"""Download the 40 ERA5 NetCDFs that the benchmark consumes.

Pulls 5 variables x 8 years (2018-2025) of hourly reanalysis data over
Turkey directly from Copernicus CDS, mirroring what previously lived in
data/raw/ before that directory was removed from git history.

One-time setup
--------------
1. Register a free account at https://cds.climate.copernicus.eu/
2. Accept the ERA5 dataset terms (one-time click on the dataset page).
3. Save your CDS API key to ~/.cdsapirc (on Windows: %USERPROFILE%/.cdsapirc):

   url: https://cds.climate.copernicus.eu/api/v2
   key: YOUR_UID:YOUR_API_KEY

   The credentials are visible at https://cds.climate.copernicus.eu/api-how-to

4. Install the client:
   pip install cdsapi

Run
---
   python scripts/download_era5.py

   # Resume after an interruption (skips files already on disk):
   python scripts/download_era5.py --skip-existing

   # Fetch a single variable or year:
   python scripts/download_era5.py --variables t2m
   python scripts/download_era5.py --years 2024 2025

Approximate wall-clock: 30-90 minutes depending on CDS queue depth.
Each file is ~45 MB; total ~1.7 GB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Maps short names used in our filenames to CDS ERA5 variable identifiers.
VARIABLES = {
    "t2m":  "2m_temperature",
    "d2m":  "2m_dewpoint_temperature",
    "u10":  "10m_u_component_of_wind",
    "v10":  "10m_v_component_of_wind",
    "ssrd": "surface_solar_radiation_downwards",
}

YEARS = [str(y) for y in range(2018, 2026)]  # 2018..2025 inclusive

# Turkey bounding box used by the v2 pipeline (matches src/data/spatial_weights.py).
# Order is [north, west, south, east] in CDS API convention.
TURKEY_BBOX = [42.5, 25.5, 35.5, 45.0]

OUT_DIR = Path("data/raw")


def fetch_one(client, short: str, long: str, year: str, out_path: Path) -> None:
    """Request one (variable, year) cube from CDS and stream it to disk."""
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": long,
            "year": year,
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": TURKEY_BBOX,
            "format": "netcdf",
        },
        str(out_path),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variables", nargs="+", choices=sorted(VARIABLES),
                    default=sorted(VARIABLES), help="Subset of variables to fetch.")
    ap.add_argument("--years", nargs="+", default=YEARS,
                    help="Subset of years (default: 2018-2025).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip files that already exist on disk.")
    args = ap.parse_args()

    try:
        import cdsapi
    except ImportError:
        print("ERROR: cdsapi not installed. Run: pip install cdsapi", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    targets = [
        (short, VARIABLES[short], year)
        for short in args.variables
        for year in args.years
    ]
    print(f"Fetching {len(targets)} (variable, year) cubes -> {OUT_DIR}/")

    for i, (short, long, year) in enumerate(targets, 1):
        out_path = OUT_DIR / f"{short}_{year}.nc"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [{i:2d}/{len(targets)}] skip {out_path.name} (exists, {out_path.stat().st_size:,} bytes)")
            continue
        print(f"  [{i:2d}/{len(targets)}] {short:>4} {year} -> {out_path.name}", flush=True)
        try:
            fetch_one(client, short, long, year, out_path)
        except Exception as e:
            print(f"    FAILED: {e}", file=sys.stderr)
            return 1

    print(f"\nDone. {len(targets)} files in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
