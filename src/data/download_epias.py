from eptr2 import EPTR2
import pandas as pd
from pathlib import Path

eptr = EPTR2(use_dotenv=True, recycle_tgt=True)

rt_cons = []
for year in range(2018, 2026):
    print(f"Fetching {year}...")

    df = eptr.call("rt-cons",start_date=f"{year}-01-01",end_date=f"{year}-12-31") # real time consumption

    rt_cons.append(df)

# Merge everything
df = pd.concat(rt_cons, ignore_index=True)

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(exist_ok=True)

df.to_csv(DATA_DIR /"electricity_consumption_2018_2025.csv", index=False)

