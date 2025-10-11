#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
import csv
import json
import glob
import argparse
import logging
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS (DK NHL Classic)
# ───────────────────────────────────────────────────────────────────────────────
DK_SALARY_CAP = 50000
DK_ROSTER = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
NEED = {"C":2,"W":3,"D":2,"G":1}  # skater UTIL added after

ELIGIBILITY = {
    "C": ["C1", "C2", "UTIL"],
    "W": ["W1", "W2", "W3", "UTIL"],
    "D": ["D1", "D2", "UTIL"],
    "G": ["G"],
}

DEFAULT_RANDOMNESS_BY_POS = {"C": 8.0, "W": 12.0, "D": 10.0, "G": 0.0}

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

LOG = logging.getLogger("nhl_opt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ───────────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────────
def _safe_str(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and np.isnan(x): return ""
    return str(x)

def _norm_name(x) -> str:
    s = _safe_str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s).strip()
    return s

def _pid_key(name: Optional[str], team: Optional[str], pos: Optional[str]) -> str:
    return f"{_norm_name(name)}|{_safe_str(team).strip().upper()}|{_safe_str(pos).strip().upper()}"

def load_player_ids(path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    if not os.path.exists(path):
        LOG.warning("player_ids.csv not found at %s — ID decoration will be empty.", path)
        return mp
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = _safe_str(row.get("player_id", "")).strip()
            name = row.get("name", "")
            team = row.get("team", "")
            pos = row.get("pos", "")
            if pid:
                mp[_pid_key(name, team, pos)] = pid
    return mp

def decorate(name: str, team: str, pos: str, mp: Dict[str, str]) -> str:
    pid = mp.get(_pid_key(name, team, pos), "")
    return f"{_safe_str(name)} ({pid})"

def _resolve_four_paths(labs_dir: str, date: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p, patt in LABS_PATTERNS.items():
        guess = os.path.join(labs_dir, patt.format(date=date))
        if os.path.exists(guess):
            out[p] = guess
            continue
        g = glob.glob(os.path.join(labs_dir, f"fantasylabs_player_data_NHL_{p}_*.csv"))
        if not g:
            raise FileNotFoundError(f"Missing Labs file for {p}. Tried {guess}")
        out[p] = sorted(g)[-1]
    return out

def _read_labs_one(path: str, pos: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {k:v for k,v in COLMAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    for need in ["name","team","salary"]:
        if need not in df.columns:
            raise ValueError(f"{path}: missing '{need}' after rename")
    # sanitize
    df["name"] = df["name"].apply(_safe_str)
    df["team"] = df["team"].apply(_safe_str)
    if "opp" in df.columns: df["opp"] = df["opp"].apply(_safe_str)
    # salary
    df["salary"] = (df["salary"].apply(_safe_str).str.replace(",","",regex=False)
                    .str.extract(r"(\d+)", expand=False).fillna("0").astype(int))
    if "proj_points" not in df.columns: df["proj_points"] = 0.0
    df["proj_points"] = pd.to_numeric(df["proj_points"], errors="coerce").fillna(0.0)
    if "projected_own" not in df.columns: df["projected_own"] = 0.0
    df["projected_own"] = pd.to_numeric(df["projected_own"], errors="coerce").fillna(0.0)
    df["pos"] = pos
    df["name_key"] = df["name"].map(_norm_name)
    df["team"] = df["team"].str.upper().str.strip()
    if "opp" in df.columns: df["opp"] = df["opp"].str.upper().str.strip()
    return df

def load_labs_merged(labs_dir: str, date: str) -> pd.DataFrame:
    paths = _resolve_four_paths(labs_dir, date)
    frames = [_read_labs_one(pth, pos) for pos,pth in paths.items()]
    full = pd.concat(frames, ignore_index=True)
    full = full[full["name"].str.len() > 0].copy()
    # drop dup per name_key+pos, keep best proj
    full = full.sort_values(["name_key","pos","proj_points"], ascending=[True,True,False])
    full = full.drop_duplicates(subset=["name_key","pos"], keep="first").reset_index(drop=True)
    return full

def apply_randomness(df: pd.DataFrame, randomness_by_pos: Dict[str,float] | None) -> pd.DataFrame:
    rbp = randomness_by_pos or DEFAULT_RANDOMNESS_BY_POS
    out = df.copy()
    base = pd.to_numeric(out["proj_points"], errors="coerce").fillna(0.0).astype(float).values
    pos  = out["pos"].astype(str).values
    jittered = []
    for i in range(len(out)):
        pct = float(rbp.get(pos[i], 0.0))
        eps = (np.random.rand() * 2.0 - 1.0) * (pct / 100.0)
        jittered.append(base[i] * (1.0 + eps))
    out["proj_points_rand"] = jittered
    return out

# ───────────────────────────────────────────────────────────────────────────────
# FEASIBILITY DIAGNOSTICS
# ───────────────────────────────────────────────────────────────────────────────
def _approx_feasible_salary_range(pool: pd.DataFrame) -> tuple[int,int,bool,Dict[str,int]]:
    ok = True
    feas_min = 0
    feas_max = 0
    counts = {}
    for p,n in NEED.items():
        pp = pool[pool["pos"]==p].sort_values("salary")
        counts[p] = len(pp)
        if len(pp) < n:
            ok = False
        feas_min += (pp["salary"].iloc[:min(len(pp),n)].sum() if len(pp)>=n else 10**9)
        feas_max += (pp["salary"].sort_values(ascending=False).iloc[:min(len(pp),n)].sum() if len(pp)>=n else 0)
    # UTIL (skater)
    sk = pool[pool["pos"].isin(["C","W","D"])].sort_values("salary")
    if len(sk)==0: ok=False
    feas_min += (sk["salary"].iloc[0] if len(sk)>0 else 10**9)
    feas_max += (sk["salary"].iloc[-1] if len(sk)>0 else 0)
    return feas_min, feas_max, ok, counts

# ───────────────────────────────────────────────────────────────────────────────
# SIMPLE GREEDY + SALARY STEERING
# ───────────────────────────────────────────────────────────────────────────────
def _choose(pool, pos, chosen, remain_cap):
    cand = pool[(pool["pos"]==pos) & (~pool.index.isin(chosen)) & (pool["salary"]<=remain_cap)]
    if cand.empty: return None
    return int((cand["proj_points_rand"] + np.random.rand(len(cand))*1e-6).idxmax())

def _try_salary_steer(pool, roster_idx: Dict[str,int], target_min: int, current_salary: int, cap: int):
    if current_salary >= target_min:
        return roster_idx, current_salary
    order = ["UTIL","W3","W2","W1","D2","D1","C2","C1"]
    for slot in order:
        i = roster_idx.get(slot)
        if i is None: continue
        row = pool.loc[i]
        pos = row["pos"]
        remain = cap - (current_salary - int(row["salary"]))
        cand = pool[(pool["pos"]==pos) & (pool.index!=i) & (pool["salary"]<=remain)]
        cand = cand[cand["salary"] > int(row["salary"])]
        if cand.empty: continue
        j = int((cand["proj_points_rand"] + np.random.rand(len(cand))*1e-6).idxmax())
        delta = int(pool.loc[j,"salary"]) - int(row["salary"])
        roster_idx[slot] = j
        current_salary += delta
        if current_salary >= target_min:
            break
    return roster_idx, current_salary

def build_lineups(pool: pd.DataFrame, num: int, min_salary: int, max_salary: int, attempts_multiplier: int = 600) -> List[Dict[str,int]]:
    lineups: List[Dict[str,int]] = []
    attempts = 0
    max_attempts = max(1000, num * attempts_multiplier)

    while len(lineups) < num and attempts < max_attempts:
        attempts += 1
        chosen_idx: List[int] = []
        r: Dict[str,int] = {}
        sal = 0

        gi = _choose(pool, "G", chosen_idx, max_salary - sal)
        if gi is None: continue
        r["G"] = gi; chosen_idx.append(gi); sal += int(pool.loc[gi,"salary"])

        for s in ["C1","C2"]:
            i = _choose(pool, "C", chosen_idx, max_salary - sal)
            if i is None: r = {}; break
            r[s] = i; chosen_idx.append(i); sal += int(pool.loc[i,"salary"])
        if not r: continue

        for s in ["D1","D2"]:
            i = _choose(pool, "D", chosen_idx, max_salary - sal)
            if i is None: r = {}; break
            r[s] = i; chosen_idx.append(i); sal += int(pool.loc[i,"salary"])
        if not r: continue

        for s in ["W1","W2","W3"]:
            i = _choose(pool, "W", chosen_idx, max_salary - sal)
            if i is None: r = {}; break
            r[s] = i; chosen_idx.append(i); sal += int(pool.loc[i,"salary"])
        if not r: continue

        util_pool = pool[(pool["pos"].isin(["C","W","D"])) & (~pool.index.isin(chosen_idx)) & (pool["salary"]<= (max_salary - sal))]
        if util_pool.empty: continue
        util_i = int((util_pool["proj_points_rand"] + np.random.rand(len(util_pool))*1e-6).idxmax())
        r["UTIL"] = util_i; chosen_idx.append(util_i); sal += int(pool.loc[util_i,"salary"])

        if sal < min_salary:
            r, sal = _try_salary_steer(pool, r, min_salary, sal, max_salary)

        if sal < min_salary or sal > max_salary:
            continue

        name_set = tuple(sorted(pool.loc[list(r.values()), "name_key"].tolist()))
        if any(tuple(sorted(pool.loc[list(x.values()), "name_key"].tolist())) == name_set for x in lineups):
            continue
        lineups.append(r)

    return lineups

# ───────────────────────────────────────────────────────────────────────────────
# EXPORT / CLI
# ───────────────────────────────────────────────────────────────────────────────
def export_lineups(pool: pd.DataFrame, lineups: List[Dict[str,int]], player_ids: Dict[str,str], out_path: str, raw_path: Optional[str]=None) -> None:
    rows, rows_raw = [], []
    for r in lineups:
        def cell(idx: Optional[int]) -> str:
            if idx is None: return ""
            row = pool.loc[idx]
            return decorate(row["name"], row.get("team",""), row.get("pos",""), player_ids)
        row = {
            "C1": cell(r.get("C1")), "C2": cell(r.get("C2")),
            "W1": cell(r.get("W1")), "W2": cell(r.get("W2")), "W3": cell(r.get("W3")),
            "D1": cell(r.get("D1")), "D2": cell(r.get("D2")),
            "G":  cell(r.get("G")),  "UTIL": cell(r.get("UTIL")),
        }
        idxs = [i for i in r.values() if i is not None]
        row["TotalSalary"] = int(pool.loc[idxs,"salary"].sum())
        row["ProjPoints"]  = round(float(pool.loc[idxs,"proj_points_rand"].sum()), 3)
        rows.append(row)
        rows_raw.append({k: (pool.loc[r[k], "name"] if r.get(k) is not None else "") for k in ["C1","C2","W1","W2","W3","D1","D2","G","UTIL"]})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if raw_path:
        pd.DataFrame(rows_raw).to_csv(raw_path, index=False)
    LOG.info("Wrote %s%s", out_path, f" and {raw_path}" if raw_path else "")

def _load_config(path: Optional[str]) -> Dict:
    if not path:
        for c in ("config.json","sample.config.json"):
            if os.path.exists(c):
                with open(c,"r",encoding="utf-8") as f: return json.load(f)
        return {}
    if path.lower().endswith((".yaml",".yml")):
        import yaml
        with open(path,"r",encoding="utf-8") as f: return yaml.safe_load(f) or {}
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def parse_args():
    ap = argparse.ArgumentParser("NHL Optimizer (robust)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--labs-dir", default="dk_data")
    ap.add_argument("--out", default="out/lineups_{date}.csv")
    ap.add_argument("--export-raw", action="store_true")
    ap.add_argument("--num-lineups", type=int, default=150)
    ap.add_argument("--min-salary", type=int, default=48000)
    ap.add_argument("--max-salary", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    cfg = _load_config(args.config)
    date = args.date
    labs_dir = args.labs_dir
    out_path = (args.out or "out/lineups_{date}.csv").format(date=date)
    raw_path = out_path.replace(".csv","_raw.csv") if args.export_raw else None

    df = load_labs_merged(labs_dir, date)
    rbp = cfg.get("randomness_pct_by_pos", DEFAULT_RANDOMNESS_BY_POS)
    df = apply_randomness(df, rbp)

    # Feasibility check with clear logs
    feas_min, feas_max, ok, counts = _approx_feasible_salary_range(df)
    LOG.info("Pool counts: %s", counts)
    LOG.info("Approx feasible salary range: min=%d, max=%d | requested [%d, %d]",
             feas_min, feas_max, args.min_salary, args.max_salary)
    if not ok:
        LOG.error("Pool incomplete by position → infeasible. Ensure C(2), W(3), D(2), G(1) are available.")
        sys.exit(2)
    if feas_min > args.max_salary:
        LOG.error("Cheapest valid roster exceeds max_salary → raise cap or use cheaper pool.")
        sys.exit(2)
    if feas_max < args.min_salary:
        LOG.error("Most expensive valid roster < min_salary → lower min_salary.")
        sys.exit(2)

    lineups = build_lineups(
        df,
        num=int(cfg.get("num_lineups", args.num_lineups)),
        min_salary=int(cfg.get("min_salary", args.min_salary)),
        max_salary=int(cfg.get("max_salary", args.max_salary)),
        attempts_multiplier=int(cfg.get("attempts_multiplier", 800)),
    )
    if not lineups:
        LOG.error("No lineups produced — try --min-salary 45000 or set config.attempts_multiplier to 1200; also verify position counts.")
        sys.exit(2)

    pid_map = load_player_ids(os.path.join("dk_data", "player_ids.csv"))
    export_lineups(df, lineups, pid_map, out_path, raw_path=raw_path)

if __name__ == "__main__":
    main()
