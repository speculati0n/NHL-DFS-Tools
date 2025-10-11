#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
import json
import glob
import argparse
import logging
import unicodedata
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import pulp
except Exception:  # pragma: no cover - allow running without ILP fallback
    pulp = None

# Optional DK export mapping support (graceful if missing)
try:
    from nhl_tools.id_mapping import load_player_ids_any, find_pid
except Exception:
    load_player_ids_any = None
    find_pid = None

DK_SALARY_CAP = 50000
DK_ROSTER = ["C1","C2","W1","W2","W3","D1","D2","G","UTIL"]
NEED = {"C":2,"W":3,"D":2,"G":1}
DEFAULT_RANDOMNESS_BY_POS = {"C":8.0,"W":12.0,"D":10.0,"G":0.0}

LABS_PATTERNS = {
    "C": "fantasylabs_player_data_NHL_C_{date}.csv",
    "W": "fantasylabs_player_data_NHL_W_{date}.csv",
    "D": "fantasylabs_player_data_NHL_D_{date}.csv",
    "G": "fantasylabs_player_data_NHL_G_{date}.csv",
}

LOG = logging.getLogger("nhl_opt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def _safe_str(x) -> str:
    if x is None: return ""
    if isinstance(x, float) and np.isnan(x): return ""
    return str(x)

def _norm_name(x) -> str:
    s = _safe_str(x)
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s).strip()
    return s

def _resolve_four_paths(labs_dir: str, date: str) -> Dict[str,str]:
    out: Dict[str,str] = {}
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

# -------- header normalization for C/W/D
def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        cl = c.strip()
        low = cl.lower()
        if low in ("player","name"): ren[c] = "name"
        elif low in ("team","teamabbrev","team_abbrev","teamabbr"): ren[c] = "team"
        elif low in ("opp","opponent"): ren[c] = "opp"
        elif low.replace(" ","") in ("salary","dksalary","dk_salary"): ren[c] = "salary"
        elif low in ("proj","projection","projected","proj_points","fpts","points","median projection","median_projection","medianprojection"):
            ren[c] = "proj_points"
        elif low in ("own","ownership","proj_own","projected_own"): ren[c] = "projected_own"
        elif low in ("pos","position","roster position","roster_position"): ren[c] = "pos_raw"
    return df.rename(columns=ren)

def _clean_frame(df: pd.DataFrame, forced_pos: str) -> pd.DataFrame:
    x = df.copy()
    for need in ["name","team","salary"]:
        if need not in x.columns: x[need] = ""
    x["name"] = x["name"].apply(_safe_str)
    x["team"] = x["team"].apply(_safe_str)
    x["salary"] = (
        x["salary"].apply(_safe_str)
        .str.replace(",","", regex=False)
        .str.replace("$","", regex=False)
        .str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    )
    if "proj_points" not in x.columns: x["proj_points"] = 0.0
    x["proj_points"] = pd.to_numeric(x["proj_points"], errors="coerce").fillna(0.0)
    x["pos"] = forced_pos
    x["team"] = x["team"].str.upper().str.strip()
    x = x[["name","team","salary","proj_points","pos"]]
    x = x[(x["name"].str.strip()!="") & (x["salary"]>0)].copy()
    return x

# -------- ALWAYS strict parser for G (by index)
def _read_goalies_strict(path: str) -> pd.DataFrame:
    """
    Split each data line on commas and take fixed columns by index:
      idx1=Player, idx2=Salary, idx6=Team, (optional) idx8=Proj
    Matches your 'normal' FantasyLabs goalie export.
    """
    rows: List[Dict[str,str]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()  # header
        for ln in f:
            parts = ln.rstrip("\n").split(",")
            if len(parts) < 7:  # need up to index 6 for Team
                continue
            player = parts[1].strip()
            salary = parts[2].strip()
            team   = parts[6].strip()
            proj = 0.0
            if len(parts) > 8:
                m = re.search(r"[-+]?\d*\.?\d+", parts[8])
                if m:
                    try: proj = float(m.group(0))
                    except Exception: proj = 0.0
            if not player or not salary or not team:
                continue
            rows.append({"name": player, "team": team, "salary": salary, "proj_points": proj, "pos": "G"})
    if not rows:
        return pd.DataFrame(columns=["name","team","salary","proj_points","pos"])
    df = pd.DataFrame(rows)
    df["salary"] = (
        df["salary"].astype(str).str.replace(",","", regex=False).str.replace("$","", regex=False)
        .str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    )
    df["proj_points"] = pd.to_numeric(df["proj_points"], errors="coerce").fillna(0.0)
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df = df[(df["name"].astype(str).str.strip()!="") & (df["salary"]>0)]
    return df[["name","team","salary","proj_points","pos"]].copy()

def _read_labs_one(path: str, pos: str) -> pd.DataFrame:
    # FORCE strict parser for goalies
    if pos == "G":
        LOG.info("Using strict index parser for %s", os.path.basename(path))
        return _read_goalies_strict(path)

    # C/W/D: normal robust mapping
    df = pd.read_csv(path, engine="python")
    LOG.info("Reading %s (%s) with columns: %s", os.path.basename(path), pos, list(df.columns))
    df = _normalize_headers(df)
    return _clean_frame(df, pos)

def load_labs_merged(labs_dir: str, date: str) -> pd.DataFrame:
    paths = _resolve_four_paths(labs_dir, date)
    frames = []
    for pos, pth in paths.items():
        frames.append(_read_labs_one(pth, pos))
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["name","team","salary","proj_points","pos"]
    )
    full = full[full["name"].astype(str).str.len()>0].copy()
    full["name_key"] = full["name"].map(_norm_name)
    full = full.sort_values(["name_key","pos","proj_points"], ascending=[True,True,False])
    full = full.drop_duplicates(subset=["name_key","pos"], keep="first").reset_index(drop=True)
    return full

def apply_randomness(df: pd.DataFrame, randomness_by_pos: Dict[str,float] | None) -> pd.DataFrame:
    rbp = randomness_by_pos or DEFAULT_RANDOMNESS_BY_POS
    out = df.copy()
    base = pd.to_numeric(out["proj_points"], errors="coerce").fillna(0.0).astype(float).values
    pos  = out["pos"].astype(str).values
    out["proj_points_rand"] = [
        base[i] * (1.0 + ((np.random.rand()*2 - 1) * (float(rbp.get(pos[i],0.0))/100.0)))
        for i in range(len(out))
    ]
    return out

def _approx_feasible_salary_range(pool: pd.DataFrame) -> tuple[int,int,bool,Dict[str,int]]:
    ok = True
    feas_min = 0
    feas_max = 0
    counts: Dict[str,int] = {}
    for p,n in NEED.items():
        pp = pool[pool["pos"]==p].sort_values("salary")
        counts[p] = len(pp)
        if len(pp) < n: ok = False
        feas_min += (pp["salary"].iloc[:min(len(pp),n)].sum() if len(pp)>=n else 10**9)
        feas_max += (pp["salary"].sort_values(ascending=False).iloc[:min(len(pp),n)].sum() if len(pp)>=n else 0)
    sk = pool[pool["pos"].isin(["C","W","D"])].sort_values("salary")
    if len(sk)==0:
        ok = False
        util_min = 10**9; util_max = 0
    else:
        util_min = int(sk["salary"].iloc[0]); util_max = int(sk["salary"].iloc[-1])
        feas_min += util_min; feas_max += util_max
    return feas_min, feas_max, ok, counts

def _choose(pool, pos, chosen, remain_cap):
    cand = pool[(pool["pos"] == pos) & (~pool.index.isin(chosen)) & (pool["salary"] <= remain_cap)]
    if cand.empty:
        return None
    return int((cand["proj_points_rand"] + np.random.rand(len(cand)) * 1e-6).idxmax())

def _try_salary_steer(pool, roster_idx: Dict[str,int], target_min: int, current_salary: int, cap: int):
    """
    Attempt to increase lineup salary to meet ``target_min`` while staying under ``cap``.

    The old heuristic only made one pass through the roster, swapping in a
    slightly higher-salary player whenever possible.  With very wide salary
    distributions this left thousands of dollars unused, which in turn caused
    the optimizer to fail whenever a user requested a tight salary floor (e.g.
    48k-50k).

    We now construct a small set of candidate upgrades for each roster slot and
    search combinations of one, two, or three simultaneous replacements.  This
    effectively solves a tiny knapsack problem and reliably finds an upgrade
    that meets the minimum salary (if one exists) without exploding the search
    space.
    """

    if current_salary >= target_min:
        return roster_idx, current_salary

    MAX_CANDIDATES_PER_SLOT = 25
    order = ["UTIL", "G", "W3", "W2", "W1", "D2", "D1", "C2", "C1"]

    used = set(roster_idx.values())
    current_proj = float(pool.loc[list(roster_idx.values()), "proj_points_rand"].sum())

    slot_candidates: Dict[str, List[tuple[int, int, float]]] = {}
    slot_delta_cap: Dict[str, int] = {}
    for slot in order:
        idx = roster_idx.get(slot)
        if idx is None:
            continue

        row = pool.loc[idx]
        base_salary = int(row["salary"])
        remain = cap - (current_salary - base_salary)

        cand = pool[(pool["pos"] == row["pos"]) & (~pool.index.isin(used - {idx})) & (pool["salary"] <= remain)]
        cand = cand[cand["salary"] > base_salary].copy()
        if cand.empty:
            continue

        cand = cand.sort_values(["salary", "proj_points_rand"], ascending=[False, False]).head(MAX_CANDIDATES_PER_SLOT)
        slot_candidates[slot] = [
            (int(i), int(pool.loc[i, "salary"]) - base_salary, float(pool.loc[i, "proj_points_rand"]) - float(row["proj_points_rand"]))
            for i in cand.index
        ]
        slot_delta_cap[slot] = max(d for _, d, _ in slot_candidates[slot]) if slot_candidates[slot] else 0

    if not slot_candidates:
        return roster_idx, current_salary

    slots = [slot for slot in order if slot in slot_candidates]
    if not slots:
        return roster_idx, current_salary

    suffix_max_delta = [0] * (len(slots) + 1)
    for i in range(len(slots) - 1, -1, -1):
        suffix_max_delta[i] = suffix_max_delta[i + 1] + slot_delta_cap.get(slots[i], 0)

    best_plan: Optional[List[tuple[str, tuple[int, int, float]]]] = None
    best_salary = current_salary
    best_proj = current_proj

    def dfs(idx: int, salary: int, proj: float, plan: List[tuple[str, tuple[int, int, float]]], used_players: set[int]) -> None:
        nonlocal best_plan, best_salary, best_proj

        if salary >= target_min:
            if best_plan is None or proj > best_proj or (np.isclose(proj, best_proj) and salary > best_salary):
                best_plan = list(plan)
                best_salary = salary
                best_proj = proj
            return

        if idx >= len(slots):
            return

        remaining_cap_room = suffix_max_delta[idx]
        if salary + remaining_cap_room < target_min:
            return

        slot = slots[idx]

        # Option: skip replacing this slot.
        dfs(idx + 1, salary, proj, plan, used_players)

        for cand in slot_candidates[slot]:
            cand_idx, delta_sal, delta_proj = cand
            if cand_idx in used_players:
                continue
            new_salary = salary + delta_sal
            if new_salary > cap:
                continue
            if new_salary + suffix_max_delta[idx + 1] < target_min:
                continue
            used_players.add(cand_idx)
            plan.append((slot, cand))
            dfs(idx + 1, new_salary, proj + delta_proj, plan, used_players)
            plan.pop()
            used_players.remove(cand_idx)

    dfs(0, current_salary, current_proj, [], set())

    if best_plan is None:
        return roster_idx, current_salary

    for slot, (idx, delta_sal, _) in best_plan:
        old_idx = roster_idx.get(slot)
        if old_idx is not None:
            roster_idx[slot] = idx
            used.discard(old_idx)
            used.add(idx)
    current_salary = best_salary

    return roster_idx, current_salary


def _assign_slots(lineup_indices: List[int], pool: pd.DataFrame) -> Dict[str, int]:
    slots: Dict[str, int] = {}
    leftovers: List[int] = []

    def take(position: str, required_slots: List[str]) -> None:
        players = [i for i in lineup_indices if pool.loc[i, "pos"] == position]
        players.sort(
            key=lambda j: (float(pool.loc[j, "proj_points_rand"]), int(pool.loc[j, "salary"])),
            reverse=True,
        )
        for slot in required_slots:
            if not players:
                raise ValueError(f"Insufficient {position} players for slot {slot}")
            slots[slot] = players.pop(0)
        leftovers.extend(players)

    take("G", ["G"])
    take("C", ["C1", "C2"])
    take("W", ["W1", "W2", "W3"])
    take("D", ["D1", "D2"])

    util_pool = [i for i in leftovers if pool.loc[i, "pos"] in {"C", "W", "D"}]
    util_pool.sort(
        key=lambda j: (float(pool.loc[j, "proj_points_rand"]), int(pool.loc[j, "salary"])),
        reverse=True,
    )
    if len(util_pool) != 1:
        raise ValueError("Expected exactly one skater for the UTIL slot")
    slots["UTIL"] = util_pool[0]
    return slots


def build_lineups(pool: pd.DataFrame, num: int, min_salary: int, max_salary: int, attempts_multiplier: int = 600) -> List[Dict[str,int]]:
    if pulp is None:
        LOG.error("PuLP solver not available; cannot build lineups deterministically.")
        return []

    players = list(pool.index)
    salary = pool["salary"].astype(int)
    proj = pool["proj_points_rand"].astype(float)
    pos_series = pool["pos"].astype(str)

    lineups: List[Dict[str, int]] = []
    used_name_sets: set[tuple[str, ...]] = set()
    previous_lineups: List[set[int]] = []

    while len(lineups) < num:
        prob = pulp.LpProblem("nhl_lineup", pulp.LpMaximize)
        x: Dict[int, pulp.LpVariable] = {
            i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in players
        }

        noise = np.random.rand(len(players)) * 1e-4
        prob += pulp.lpSum((proj.loc[i] + noise[idx]) * x[i] for idx, i in enumerate(players))

        prob += pulp.lpSum(salary.loc[i] * x[i] for i in players) <= max_salary
        prob += pulp.lpSum(salary.loc[i] * x[i] for i in players) >= min_salary
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "C") >= 2
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "C") <= 3
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "W") >= 3
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "W") <= 4
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "D") >= 2
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "D") <= 3
        prob += pulp.lpSum(x[i] for i in players if pos_series.loc[i] == "G") == 1
        prob += pulp.lpSum(x[i] for i in players) == 9

        for prev in previous_lineups:
            prob += pulp.lpSum(x[i] for i in prev) <= 8

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            break

        lineup_indices = [i for i in players if x[i].value() is not None and x[i].value() > 0.5]
        if len(lineup_indices) != 9:
            break

        sal_total = int(salary.loc[lineup_indices].sum())
        if sal_total < min_salary or sal_total > max_salary:
            previous_lineups.append(set(lineup_indices))
            continue

        name_set = tuple(sorted(pool.loc[lineup_indices, "name_key"].tolist()))
        if name_set in used_name_sets:
            previous_lineups.append(set(lineup_indices))
            continue

        try:
            roster = _assign_slots(lineup_indices, pool)
        except ValueError:
            previous_lineups.append(set(lineup_indices))
            continue

        lineups.append(roster)
        used_name_sets.add(name_set)
        previous_lineups.append(set(lineup_indices))

    if len(lineups) < num:
        LOG.warning("Generated %d lineups out of requested %d using ILP fallback", len(lineups), num)

    return lineups

def _decorate(name: str, team: str, pos: str, pid_map: Dict[str,str]) -> str:
    if find_pid and pid_map: pid = find_pid(name, team, pos, pid_map)
    else: pid = ""
    return f"{_safe_str(name)} ({pid})"

def export_lineups(pool: pd.DataFrame, lineups: List[Dict[str,int]], pid_map: Dict[str,str], out_path: str, raw_path: Optional[str]=None) -> None:
    rows, rows_raw = [], []
    for r in lineups:
        def cell(idx: Optional[int]) -> str:
            if idx is None: return ""
            row = pool.loc[idx]
            return _decorate(row["name"], row.get("team",""), row.get("pos",""), pid_map)
        row = {"C1":cell(r.get("C1")), "C2":cell(r.get("C2")),
               "W1":cell(r.get("W1")), "W2":cell(r.get("W2")), "W3":cell(r.get("W3")),
               "D1":cell(r.get("D1")), "D2":cell(r.get("D2")),
               "G":cell(r.get("G")), "UTIL":cell(r.get("UTIL"))}
        idxs = [i for i in r.values() if i is not None]
        row["TotalSalary"] = int(pool.loc[idxs,"salary"].sum())
        row["ProjPoints"]  = round(float(pool.loc[idxs,"proj_points_rand"].sum()), 3)
        rows.append(row)
        rows_raw.append({k: (pool.loc[r[k], "name"] if r.get(k) is not None else "") for k in ["C1","C2","W1","W2","W3","D1","D2","G","UTIL"]})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if raw_path: pd.DataFrame(rows_raw).to_csv(raw_path, index=False)
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
    ap = argparse.ArgumentParser("NHL Optimizer (strict goalie parser)")
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
    if args.verbose: LOG.setLevel(logging.DEBUG)

    cfg = _load_config(args.config)
    date = args.date
    labs_dir = args.labs_dir
    out_path = (args.out or "out/lineups_{date}.csv").format(date=date)
    raw_path = out_path.replace(".csv","_raw.csv") if args.export_raw else None

    df = load_labs_merged(labs_dir, date)
    counts = df.groupby("pos")["name"].count().to_dict() if not df.empty else {}
    LOG.info("Pool counts: %s", counts)

    df = apply_randomness(df, cfg.get("randomness_pct_by_pos", DEFAULT_RANDOMNESS_BY_POS))

    feas_min, feas_max, ok, _ = _approx_feasible_salary_range(df)
    LOG.info("Approx feasible salary range: min=%d, max=%d | requested [%d, %d]", feas_min, feas_max, args.min_salary, args.max_salary)
    if not ok: sys.exit(2)
    if feas_min > args.max_salary: sys.exit(2)
    if feas_max < args.min_salary: sys.exit(2)

    lineups = build_lineups(
        df,
        num=int(cfg.get("num_lineups", args.num_lineups)),
        min_salary=int(cfg.get("min_salary", args.min_salary)),
        max_salary=int(cfg.get("max_salary", args.max_salary)),
        attempts_multiplier=int(cfg.get("attempts_multiplier", 800)),
    )
    if not lineups:
        LOG.error("No lineups produced — try --min-salary 45000 or attempts_multiplier 1200; verify counts.")
        sys.exit(2)

    pid_map: Dict[str,str] = {}
    if load_player_ids_any:
        pid_map = load_player_ids_any(os.path.join("dk_data","player_ids.csv"))

    export_lineups(df, lineups, pid_map, out_path, raw_path=raw_path)

if __name__ == "__main__":
    main()
