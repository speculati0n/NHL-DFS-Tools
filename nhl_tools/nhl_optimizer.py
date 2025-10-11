#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import sys
import csv
import json
import math
import glob
import argparse
import logging
import unicodedata
from typing import Dict, List, Optional

import pandas as pd

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS (DK NHL Classic)
# ───────────────────────────────────────────────────────────────────────────────
DK_SALARY_CAP = 50000
DK_ROSTER = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]

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
def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s).strip()
    return s


def _pid_key(name: Optional[str], team: Optional[str], pos: Optional[str]) -> str:
    return f"{_norm_name(name)}|{(team or '').strip().upper()}|{(pos or '').strip().upper()}"


def add_quality_columns(df: pd.DataFrame, ceil_mult: float = 1.6, floor_mult: float = 0.55) -> pd.DataFrame:
    """Add Ceil/Floor columns based on a simple multiplier of proj_points."""
    out = df.copy()
    base = out.get("proj_points", pd.Series([0.0] * len(out), index=out.index))
    out["Ceil"] = base.astype(float) * float(ceil_mult)
    out["Floor"] = base.astype(float) * float(floor_mult)
    return out


def load_player_ids(path: str) -> Dict[str, str]:
    """Load DK player_id mapping with NFL-like semantics."""
    mp: Dict[str, str] = {}
    if not os.path.exists(path):
        LOG.warning("player_ids.csv not found at %s — ID decoration will be empty.", path)
        return mp
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row.get("player_id", "")).strip()
            name = row.get("name", "")
            team = row.get("team", "")
            pos = row.get("pos", "")
            if pid:
                mp[_pid_key(name, team, pos)] = pid
    return mp


def decorate(name: str, team: str, pos: str, mp: Dict[str, str]) -> str:
    pid = mp.get(_pid_key(name, team, pos), "")
    return f"{name} ({pid})"


def _read_labs_one(path: str, pos: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # rename columns if present
    rename = {k: v for k, v in COLMAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # required
    for need in ["name", "team", "salary"]:
        if need not in df.columns:
            raise ValueError(f"{path}: missing required column '{need}' after rename")

    # salary clean
    df["salary"] = (
        df["salary"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )

    if "proj_points" not in df.columns:
        df["proj_points"] = 0.0
    if "projected_own" not in df.columns:
        df["projected_own"] = 0.0

    df["pos"] = pos
    df["name_key"] = df["name"].map(_norm_name)
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    if "opp" in df.columns:
        df["opp"] = df["opp"].astype(str).str.upper().str.strip()
    return df


def _resolve_four_paths(labs_dir: str, date: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p, patt in LABS_PATTERNS.items():
        guess = os.path.join(labs_dir, patt.format(date=date))
        if os.path.exists(guess):
            out[p] = guess
            continue
        # fallback glob by position, pick most recent
        g = glob.glob(os.path.join(labs_dir, f"fantasylabs_player_data_NHL_{p}_*.csv"))
        if not g:
            raise FileNotFoundError(f"Missing Labs file for {p}. Tried {guess}")
        out[p] = sorted(g)[-1]
    return out


def load_labs_merged(labs_dir: str, date: str) -> pd.DataFrame:
    paths = _resolve_four_paths(labs_dir, date)
    frames = []
    for pos, pth in paths.items():
        frames.append(_read_labs_one(pth, pos))
    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["name_key", "pos", "proj_points"], ascending=[True, True, False])
    full = full.drop_duplicates(subset=["name_key", "pos"], keep="first").reset_index(drop=True)
    return full


def apply_randomness(df: pd.DataFrame, randomness_by_pos: Dict[str, float] | None) -> pd.DataFrame:
    rbp = randomness_by_pos or DEFAULT_RANDOMNESS_BY_POS
    out = df.copy()
    jitter = []
    for _, r in out.iterrows():
        pct = float(rbp.get(r["pos"], 0.0))
        # uniform ±pct% around the base projection
        eps = (2.0 * (pd.np.random.rand() - 0.5)) * (pct / 100.0)  # avoids importing random
        jitter.append(float(r["proj_points"]) * (1.0 + eps))
    out["proj_points_rand"] = jitter
    return out


# ───────────────────────────────────────────────────────────────────────────────
# SIMPLE GREEDY OPTIMIZER (baseline; replace with ILP/CP-SAT if desired)
# ───────────────────────────────────────────────────────────────────────────────
def _slot_order() -> List[str]:
    # order matters for greedy fill
    return ["G", "C", "C", "D", "D", "W", "W", "W", "UTIL"]


def _choose_from(pool: pd.DataFrame, pos: str, chosen: List[int], remain_salary: int) -> Optional[int]:
    # choose best remaining by randomized projection
    cand = pool[(pool["pos"] == pos) & (~pool.index.isin(chosen)) & (pool["salary"] <= remain_salary)]
    if cand.empty:
        return None
    return int(cand["proj_points_rand"].astype(float).idxmax())


def build_lineups(pool: pd.DataFrame, num: int, min_salary: int, max_salary: int) -> List[Dict[str, int]]:
    lineups: List[Dict[str, int]] = []
    attempts = 0
    by_slot = _slot_order()

    # pre-sort by pos not strictly necessary; we always argmax by proj_points_rand
    while len(lineups) < num and attempts < num * 300:
        attempts += 1
        chosen_idx: List[int] = []
        roster: Dict[str, int] = {}
        salary = 0

        # G first (hard slot)
        gi = _choose_from(pool, "G", chosen_idx, max_salary - salary)
        if gi is None:
            continue
        roster["G"] = gi
        chosen_idx.append(gi)
        salary += int(pool.loc[gi, "salary"])

        # C, C
        for s in ["C1", "C2"]:
            i = _choose_from(pool, "C", chosen_idx, max_salary - salary)
            if i is None:
                roster = {}
                break
            roster[s] = i
            chosen_idx.append(i)
            salary += int(pool.loc[i, "salary"])
        if not roster:
            continue

        # D, D
        for s in ["D1", "D2"]:
            i = _choose_from(pool, "D", chosen_idx, max_salary - salary)
            if i is None:
                roster = {}
                break
            roster[s] = i
            chosen_idx.append(i)
            salary += int(pool.loc[i, "salary"])
        if not roster:
            continue

        # W, W, W
        for s in ["W1", "W2", "W3"]:
            i = _choose_from(pool, "W", chosen_idx, max_salary - salary)
            if i is None:
                roster = {}
                break
            roster[s] = i
            chosen_idx.append(i)
            salary += int(pool.loc[i, "salary"])
        if not roster:
            continue

        # UTIL (any skater)
        util_pool = pool[
            (pool["pos"].isin(["C", "W", "D"]))
            & (~pool.index.isin(chosen_idx))
            & (pool["salary"] <= (max_salary - salary))
        ]
        if util_pool.empty:
            continue
        util_i = int(util_pool["proj_points_rand"].astype(float).idxmax())
        roster["UTIL"] = util_i
        chosen_idx.append(util_i)
        salary += int(pool.loc[util_i, "salary"])

        if salary < min_salary or salary > max_salary:
            continue

        # Uniqueness: here we treat uniqueness as uniqueness of full set of names
        name_set = tuple(sorted(pool.loc[chosen_idx, "name_key"].tolist()))
        if any(tuple(sorted(pool.loc[list(r.values()), "name_key"].tolist())) == name_set for r in lineups):
            continue

        lineups.append(roster)

    return lineups


def export_lineups(
    pool: pd.DataFrame,
    lineups: List[Dict[str, int]],
    player_ids: Dict[str, str],
    out_path: str,
    raw_path: Optional[str] = None,
) -> None:
    rows = []
    rows_raw = []
    for r in lineups:
        def cell(idx: Optional[int]) -> str:
            if idx is None:
                return ""
            row = pool.loc[idx]
            return decorate(row["name"], row.get("team", ""), row.get("pos", ""), player_ids)

        row = {
            "C1": cell(r.get("C1")),
            "C2": cell(r.get("C2")),
            "W1": cell(r.get("W1")),
            "W2": cell(r.get("W2")),
            "W3": cell(r.get("W3")),
            "D1": cell(r.get("D1")),
            "D2": cell(r.get("D2")),
            "G": cell(r.get("G")),
            "UTIL": cell(r.get("UTIL")),
        }
        idxs = [i for i in r.values() if i is not None]
        tot_sal = int(pool.loc[idxs, "salary"].sum())
        proj = float(pool.loc[idxs, "proj_points_rand"].sum())
        row["TotalSalary"] = tot_sal
        row["ProjPoints"] = round(proj, 3)
        rows.append(row)

        rows_raw.append({k: (pool.loc[r[k], "name"] if r.get(k) is not None else "") for k in ["C1","C2","W1","W2","W3","D1","D2","G","UTIL"]})

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    if raw_path:
        pd.DataFrame(rows_raw).to_csv(raw_path, index=False)
    LOG.info("Wrote %s%s", out_path, f" and {raw_path}" if raw_path else "")


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────
def _load_config(path: Optional[str]) -> Dict:
    if not path:
        # NFL-like fallback: sample.config.json then config.json
        for c in ("config.json", "sample.config.json"):
            if os.path.exists(c):
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
        return {}
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    ap = argparse.ArgumentParser("NHL Optimizer (NFL-style output formatting)")
    ap.add_argument("--date", required=True, type=str, help="Slate date, e.g., 2025-10-09")
    ap.add_argument("--labs-dir", default="dk_data", type=str, help="Directory containing Labs CSVs")
    ap.add_argument("--out", default="out/lineups_{date}.csv", type=str, help="Output CSV path")
    ap.add_argument("--export-raw", action="store_true", help="Also write raw names file")
    ap.add_argument("--num-lineups", default=150, type=int)
    ap.add_argument("--min-salary", default=48000, type=int)
    ap.add_argument("--max-salary", default=DK_SALARY_CAP, type=int)
    ap.add_argument("--config", default=None, type=str, help="Optional config file (JSON/YAML)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = _load_config(args.config)

    date = args.date
    labs_dir = args.labs_dir
    out_path = (args.out or "out/lineups_{date}.csv").format(date=date)
    raw_path = out_path.replace(".csv", "_raw.csv") if args.export_raw else None

    # Load projections from four Labs files (C/W/D/G)
    df = load_labs_merged(labs_dir, date)
    df = add_quality_columns(df, ceil_mult=cfg.get("ceil_mult", 1.6), floor_mult=cfg.get("floor_mult", 0.55))

    # Randomness by position (NFL-like knob)
    rbp = cfg.get("randomness_pct_by_pos", DEFAULT_RANDOMNESS_BY_POS)
    df = apply_randomness(df, rbp)

    # Build lineups (baseline)
    num_lineups = int(cfg.get("num_lineups", args.num_lineups))
    min_salary = int(cfg.get("min_salary", args.min_salary))
    max_salary = int(cfg.get("max_salary", args.max_salary))

    lineups = build_lineups(df, num_lineups, min_salary, max_salary)
    if not lineups:
        LOG.error("No lineups produced (check salary bounds, pool size, and projections).")
        sys.exit(2)

    # Player ID decoration (NFL semantics)
    pid_map = load_player_ids(os.path.join("dk_data", "player_ids.csv"))
    export_lineups(df, lineups, pid_map, out_path, raw_path=raw_path)


if __name__ == "__main__":
    main()
