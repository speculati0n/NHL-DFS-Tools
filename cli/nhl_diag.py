#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, argparse, logging, glob
from typing import Dict
import numpy as np
import pandas as pd

LOG = logging.getLogger("nhl_diag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LABS_PATTERNS = {
    "C": "fantasylabs_player_data_NHL_C_{date}.csv",
    "W": "fantasylabs_player_data_NHL_W_{date}.csv",
    "D": "fantasylabs_player_data_NHL_D_{date}.csv",
    "G": "fantasylabs_player_data_NHL_G_{date}.csv",
}
COLMAP = {
    "Player": "name",
    "Team": "team",
    "Opp": "opp",
    "Salary": "salary",
    "Proj": "proj_points",
    "Own": "projected_own",
    "Full": "full_stack_flag",
    "PP": "powerplay_flag",
}

def _safe_str(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and np.isnan(x): return ""
    return str(x)

def _resolve(labs_dir: str, date: str) -> Dict[str,str]:
    out = {}
    for p, patt in LABS_PATTERNS.items():
        guess = os.path.join(labs_dir, patt.format(date=date))
        if os.path.exists(guess):
            out[p] = guess
        else:
            g = glob.glob(os.path.join(labs_dir, f"fantasylabs_player_data_NHL_{p}_*.csv"))
            if not g:
                raise FileNotFoundError(f"Missing Labs file for {p}. Tried {guess}")
            out[p] = sorted(g)[-1]
    return out

def _read_one(path: str, pos: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {k:v for k,v in COLMAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    for need in ["name","team","salary"]:
        if need not in df.columns:
            raise ValueError(f"{path}: missing '{need}'")
    df["name"]  = df["name"].apply(_safe_str)
    df["team"]  = df["team"].apply(_safe_str)
    df["salary"] = (df["salary"].apply(_safe_str)
                    .str.replace(",","",regex=False)
                    .str.extract(r"(\d+)", expand=False).fillna("0").astype(int))
    df["pos"] = pos
    return df

def parse_args():
    ap = argparse.ArgumentParser("NHL pool feasibility diag")
    ap.add_argument("--date", required=True)
    ap.add_argument("--labs-dir", default="dk_data")
    ap.add_argument("--min-salary", type=int, default=48000)
    ap.add_argument("--max-salary", type=int, default=50000)
    return ap.parse_args()

def main():
    a = parse_args()
    paths = _resolve(a.labs_dir, a.date)
    frames = []
    for pos, p in paths.items():
        frames.append(_read_one(p, pos))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["name"].str.len()>0].copy()

    # Position counts
    cnt = df.groupby("pos")["name"].count().to_dict()
    LOG.info("Counts per pos: %s", cnt)

    # Check min/max feasible salary by greedy cheapest / priciest pick per slot
    need = {"C":2,"W":3,"D":2,"G":1}  # UTIL later
    feas_min = 0
    feas_max = 0
    ok = True
    for p, n in need.items():
        poolp = df[df["pos"]==p].sort_values("salary")
        if len(poolp) < n:
            LOG.error("Not enough %s (need %d, have %d)", p, n, len(poolp))
            ok = False
        feas_min += poolp["salary"].iloc[:min(len(poolp), n)].sum() if len(poolp)>=n else 10**9
        feas_max += poolp["salary"].sort_values(ascending=False).iloc[:min(len(poolp), n)].sum() if len(poolp)>=n else 0

    # UTIL: best remaining skater min/max
    sk = df[df["pos"].isin(["C","W","D"])].sort_values("salary")
    if len(sk) == 0:
        LOG.error("No skaters for UTIL")
        ok = False
    else:
        feas_min += sk["salary"].iloc[0]
        feas_max += sk["salary"].iloc[-1]

    LOG.info("Feasible salary range (approx): min=%d, max=%d", feas_min, feas_max)
    LOG.info("Requested salary window: [%d, %d]", a.min_salary, a.max_salary)

    if not ok:
        LOG.error("Pool incomplete by position — infeasible.")
        sys.exit(2)

    if feas_min > a.max_salary:
        LOG.error("Even the CHEAPEST valid roster exceeds max_salary → lower salaries or raise cap.")
        sys.exit(2)
    if feas_max < a.min_salary:
        LOG.error("Even the MOST EXPENSIVE valid roster is below min_salary → lower min_salary.")
        sys.exit(2)

    LOG.info("Pool looks feasible. If opt still fails, likely selection heuristic — try lowering min-salary or increase attempts.")
    # Show top/bottom salaries per pos for quick context
    for p in ["G","C","D","W"]:
        poolp = df[df["pos"]==p]
        if not poolp.empty:
            LOG.info("%s salary min=%d, max=%d, median=%d (n=%d)",
                     p, poolp["salary"].min(), poolp["salary"].max(), int(poolp["salary"].median()), len(poolp))

if __name__ == "__main__":
    main()
