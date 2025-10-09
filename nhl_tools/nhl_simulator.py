from __future__ import annotations
import argparse, os, sys, math
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.stats import gamma, lognorm

from .nhl_data import load_labs_for_date


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LABS_DIR = os.path.join(REPO_ROOT, "dk_data")

def _sample_points(mean: float, vol: float, rng: np.random.Generator) -> float:
    """
    Simple mixture: mostly gamma around mean; volatility scales variance.
    """
    mean = max(mean, 0.1)
    sigma = max(0.15*mean, vol*mean)  # stdev
    # lognormal params
    mu = math.log(mean**2 / math.sqrt(sigma**2 + mean**2))
    s  = math.sqrt(math.log(1 + (sigma**2)/(mean**2)))
    return float(lognorm(s=s, scale=math.exp(mu)).rvs(random_state=rng))

def simulate_lineups(lineups_csv: str,
                     labs_dir: str, date: str,
                     sims: int,
                     ceil_mult: float, floor_mult: float,
                     w_up: float, w_con: float, w_dud: float,
                     seed: int = 42) -> pd.DataFrame:
    """
    For each lineup, sample player outcomes and sum DK points; report quantiles & dud rates.
    """
    base = load_labs_for_date(labs_dir, date)
    base = base.set_index(["Name","Team","PosCanon"])

    lu = pd.read_csv(lineups_csv)
    lu["key"] = list(zip(lu["Name"], lu["Team"], lu["PosCanon"]))
    # attach means
    def _m(k):
        return base.loc[k,"Proj"] if k in base.index else np.nan
    lu["Mean"] = lu["key"].map(_m)
    if lu["Mean"].isna().any():
        print("Warning: some players not found in Labs for sim; dropping those rows.")
        lu = lu.dropna(subset=["Mean"])

    # volatility proxy from upside/consistency
    lu["Ceil"] = lu["Mean"] * ceil_mult
    lu["Floor"]= lu["Mean"] * floor_mult
    g = lu.groupby("PosCanon")
    upZ = (lu["Ceil"] - g["Ceil"].transform("mean")) / (g["Ceil"].transform("std").replace(0,np.nan))
    coZ = (lu["Floor"] - g["Floor"].transform("mean")) / (g["Floor"].transform("std").replace(0,np.nan))
    lu["Vol"] = (0.30 + 0.12*w_up*upZ.abs().fillna(0) - 0.05*w_con*coZ.clip(upper=0).abs().fillna(0)).clip(0.15, 0.6)

    rng = np.random.default_rng(seed)
    L = lu["LineupID"].nunique()
    results = {lid: [] for lid in sorted(lu["LineupID"].unique())}
    for _ in range(int(sims)):
        sampled = lu.copy()
        sampled["Pts"] = sampled.apply(lambda r: _sample_points(r["Mean"], r["Vol"], rng), axis=1)
        totals = sampled.groupby("LineupID")["Pts"].sum()
        for lid, val in totals.items():
            results[lid].append(float(val))

    rows = []
    for lid, arr in results.items():
        a = np.array(arr)
        rows.append({
            "LineupID": lid,
            "SimMean": float(a.mean()),
            "P90": float(np.percentile(a, 90)),
            "P75": float(np.percentile(a, 75)),
            "P50": float(np.percentile(a, 50)),
            "P25": float(np.percentile(a, 25)),
            "Std": float(a.std(ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("P90", ascending=False)

def main():
    ap = argparse.ArgumentParser(description="NHL lineup simulator (separate from NFL).")
    ap.add_argument("--lineups", required=True)
    ap.add_argument("--labs-dir", default=DEFAULT_LABS_DIR,
                    help="Folder with FantasyLabs NHL CSVs (default: %(default)s)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--sims", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-up", type=float, default=0.15)
    ap.add_argument("--w-con", type=float, default=0.05)
    ap.add_argument("--w-dud", type=float, default=0.03)
    ap.add_argument("--ceil-mult", type=float, default=1.6)
    ap.add_argument("--floor-mult", type=float, default=0.55)
    args = ap.parse_args()

    rep = simulate_lineups(args.lineups, args.labs_dir, args.date, args.sims,
                           args.ceil_mult, args.floor_mult,
                           args.w_up, args.w_con, args.w_dud)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rep.to_csv(args.out, index=False)
    print(f"Wrote sim report -> {args.out}")

if __name__ == "__main__":
    main()
