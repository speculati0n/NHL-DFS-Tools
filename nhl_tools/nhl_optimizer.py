#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pulp import (
    LpBinary,
    LpMaximize,
    LpProblem,
    LpStatusOptimal,
    LpVariable,
    PULP_CBC_CMD,
    lpSum,
)

from .nhl_projection_loader import load_labs_for_date as load_labs_raw
from .nhl_player_ids import decorate_with_ids, load_player_ids
from .nhl_stacks import line_bucket, game_pairs

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LABS_DIR = os.path.join(REPO_ROOT, "dk_data")
DEFAULT_PLAYER_IDS = os.path.join(DEFAULT_LABS_DIR, "player_ids.csv")

DK_SALARY_CAP = 50000
DK_ROSTER_COUNTS = dict(C=2, W=3, D=2, G=1, UTIL=1)
DK_ROSTER_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
DEFAULT_RANDOMNESS = {"C": 8, "W": 12, "D": 10, "G": 0}

log = logging.getLogger("nhl_opt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _sanitize_date_for_loader(date: str) -> str:
    return date.replace("-", "")


def _prepare_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Name"] = out["name"].astype(str).str.strip()
    out["Team"] = out["team"].astype(str).str.upper().str.strip()
    out["Opp"] = out.get("opp", "").astype(str).str.upper().str.strip()
    out["Salary"] = out["salary"].astype(int)
    out["Proj"] = out["proj_points"].astype(float)
    out["ProjMean"] = out["Proj"]
    out["Own"] = out.get("projected_own", 0.0).fillna(0.0).astype(float)
    out["PosCanon"] = out["pos"].astype(str).str.upper()
    out["FullRaw"] = out.get("full_stack_flag")
    out["PP"] = out.get("powerplay_flag")
    return out


def _apply_randomness(df: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    rp = cfg.get("randomness_pct_by_pos", DEFAULT_RANDOMNESS)
    if not rp:
        return df
    seed = cfg.get("random_seed")
    rng = random.Random(seed)
    jittered: List[float] = []
    for _, row in df.iterrows():
        pct = 0.0
        if isinstance(rp, dict):
            pct = float(rp.get(row["PosCanon"], rp.get(row.get("pos"), 0.0)))
        try:
            pct = float(pct)
        except Exception:
            pct = 0.0
        if pct <= 0:
            jittered.append(float(row["Proj"]))
            continue
        noise = rng.uniform(-pct, pct) / 100.0
        jittered.append(float(row["Proj"]) * (1.0 + noise))
    df = df.copy()
    df["ProjRand"] = jittered
    df["Proj"] = df["ProjRand"]
    return df


# ---------------------------------------------------------------------------
# Quality columns and solver (legacy logic retained with NFL-style hooks)
# ---------------------------------------------------------------------------

def add_quality_columns(df: pd.DataFrame, ceil_mult: float = 1.6, floor_mult: float = 0.55) -> pd.DataFrame:
    out = df.copy()
    out["Ceil"] = out["Proj"] * float(ceil_mult)
    out["Floor"] = out["Proj"] * float(floor_mult)

    def zcol(col: str, by: str = "PosCanon"):
        g = out.groupby(by)[col]
        return (out[col] - g.transform("mean")) / (g.transform("std").replace(0, np.nan))

    out["UpsideZ"] = zcol("Ceil")
    out["ConsistencyZ"] = -zcol("Floor")
    out["DudPenalty"] = (3.0 / out["ProjMean"].clip(lower=1e-3)).clip(upper=2.0)
    return out


def solve_single_lineup(
    players: pd.DataFrame,
    w_up: float,
    w_con: float,
    w_dud: float,
    min_salary: int,
    max_salary: int,
    max_vs_goalie: int,
    need_ev2: int,
    need_ev3: int,
    need_pp1_2: int,
    need_pp1_3: int,
    need_pp2_2: int,
    need_same2: int,
    need_same3: int,
    need_same4: int,
    need_same5: int,
    need_bringback: int,
    num_uniques: int,
    history: Optional[List[List[str]]] = None,
) -> Tuple[bool, List[int]]:
    df = players.reset_index(drop=True).copy()
    N = len(df)
    idx = range(N)
    C = [i for i in idx if df.loc[i, "PosCanon"] == "C"]
    W = [i for i in idx if df.loc[i, "PosCanon"] == "W"]
    D = [i for i in idx if df.loc[i, "PosCanon"] == "D"]
    G = [i for i in idx if df.loc[i, "PosCanon"] == "G"]
    SK = [i for i in idx if i not in G]

    prob = LpProblem("DK_NHL", LpMaximize)

    x = LpVariable.dicts("x", idx, lowBound=0, upBound=1, cat=LpBinary)

    score = (
        df["Proj"]
        * (1.0 + w_up * df["UpsideZ"].fillna(0.0))
        * (1.0 + w_con * df["ConsistencyZ"].fillna(0.0))
        - w_dud * df["DudPenalty"].fillna(0.0)
    )
    prob += lpSum(x[i] * float(score.loc[i]) for i in idx)

    prob += lpSum(x[i] * int(df.loc[i, "Salary"]) for i in idx) >= int(min_salary)
    prob += lpSum(x[i] * int(df.loc[i, "Salary"]) for i in idx) <= int(max_salary)

    prob += lpSum(x[i] for i in C) >= DK_ROSTER_COUNTS["C"]
    prob += lpSum(x[i] for i in W) >= DK_ROSTER_COUNTS["W"]
    prob += lpSum(x[i] for i in D) >= DK_ROSTER_COUNTS["D"]
    prob += lpSum(x[i] for i in G) == DK_ROSTER_COUNTS["G"]
    prob += lpSum(x[i] for i in idx) == sum(DK_ROSTER_COUNTS.values())

    for team, _ in df.loc[SK, "Team"].value_counts().items():
        Ti = [i for i in SK if df.loc[i, "Team"] == team]
        prob += lpSum(x[i] for i in Ti) <= 5

    for gi in G:
        opp = df.loc[gi, "Opp"]
        if not isinstance(opp, str):
            continue
        vs = [i for i in SK if df.loc[i, "Team"] == opp]
        prob += lpSum(x[i] for i in vs) <= int(max_vs_goalie) + (1 - x[gi]) * 9

    df = line_bucket(df)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, r in df.iterrows():
        if r.get("EV_LINE_TAG"):
            groups.setdefault(("EV", r["EV_LINE_TAG"]), []).append(i)
        if r.get("PP_TAG"):
            groups.setdefault(("PP", r["PP_TAG"]), []).append(i)

    y_ev2 = [LpVariable(f"y_ev2_{k[1]}", 0, 1, LpBinary) for k in groups.keys() if k[0] == "EV"]
    ev2_keys = [k for k in groups.keys() if k[0] == "EV"]
    for y, k in zip(y_ev2, ev2_keys):
        members = groups[k]
        prob += lpSum(x[i] for i in members) >= 2 * y
    prob += lpSum(y_ev2) >= int(need_ev2)

    y_ev3 = [LpVariable(f"y_ev3_{k[1]}", 0, 1, LpBinary) for k in ev2_keys]
    for y, k in zip(y_ev3, ev2_keys):
        members = groups[k]
        prob += lpSum(x[i] for i in members) >= 3 * y
    prob += lpSum(y_ev3) >= int(need_ev3)

    pp1_keys = [k for k in groups.keys() if k[0] == "PP" and str(k[1]).endswith("PP1")]
    y_pp1_2 = [LpVariable(f"y_pp1_2_{k[1]}", 0, 1, LpBinary) for k in pp1_keys]
    y_pp1_3 = [LpVariable(f"y_pp1_3_{k[1]}", 0, 1, LpBinary) for k in pp1_keys]
    for y2, y3, k in zip(y_pp1_2, y_pp1_3, pp1_keys):
        mem = groups[k]
        prob += lpSum(x[i] for i in mem) >= 2 * y2
        prob += lpSum(x[i] for i in mem) >= 3 * y3
    prob += lpSum(y_pp1_2) >= int(need_pp1_2)
    prob += lpSum(y_pp1_3) >= int(need_pp1_3)

    pp2_keys = [k for k in groups.keys() if k[0] == "PP" and str(k[1]).endswith("PP2")]
    y_pp2_2 = [LpVariable(f"y_pp2_2_{k[1]}", 0, 1, LpBinary) for k in pp2_keys]
    for y, k in zip(y_pp2_2, pp2_keys):
        mem = groups[k]
        prob += lpSum(x[i] for i in mem) >= 2 * y
    prob += lpSum(y_pp2_2) >= int(need_pp2_2)

    teams = sorted(df["Team"].dropna().unique().tolist())
    y_same2 = [LpVariable(f"y_same2_{t}", 0, 1, LpBinary) for t in teams]
    y_same3 = [LpVariable(f"y_same3_{t}", 0, 1, LpBinary) for t in teams]
    y_same4 = [LpVariable(f"y_same4_{t}", 0, 1, LpBinary) for t in teams]
    y_same5 = [LpVariable(f"y_same5_{t}", 0, 1, LpBinary) for t in teams]
    for t, y2, y3, y4, y5 in zip(teams, y_same2, y_same3, y_same4, y_same5):
        Ti = [i for i in range(N) if (df.loc[i, "Team"] == t and df.loc[i, "PosCanon"] != "G")]
        prob += lpSum(x[i] for i in Ti) >= 2 * y2
        prob += lpSum(x[i] for i in Ti) >= 3 * y3
        prob += lpSum(x[i] for i in Ti) >= 4 * y4
        prob += lpSum(x[i] for i in Ti) >= 5 * y5
    prob += lpSum(y_same2) >= int(need_same2)
    prob += lpSum(y_same3) >= int(need_same3)
    prob += lpSum(y_same4) >= int(need_same4)
    prob += lpSum(y_same5) >= int(need_same5)

    games = game_pairs(df)
    y_bb = [LpVariable(f"y_bb_{a}_{b}", 0, 1, LpBinary) for a, b in games]
    y_stack_a = [LpVariable(f"y_stack_{a}_{b}_{a}", 0, 1, LpBinary) for a, b in games]
    y_stack_b = [LpVariable(f"y_stack_{a}_{b}_{b}", 0, 1, LpBinary) for a, b in games]
    for (a, b), y_game, ya, yb in zip(games, y_bb, y_stack_a, y_stack_b):
        A = [i for i in SK if str(df.loc[i, "Team"]).upper() == a]
        B = [i for i in SK if str(df.loc[i, "Team"]).upper() == b]
        if not A or not B:
            prob += y_game == 0
            prob += ya == 0
            prob += yb == 0
            continue
        prob += lpSum(x[i] for i in A) >= 2 * ya
        prob += lpSum(x[i] for i in B) >= ya
        prob += lpSum(x[i] for i in B) >= 2 * yb
        prob += lpSum(x[i] for i in A) >= yb
        prob += y_game <= ya + yb
        prob += y_game >= ya
        prob += y_game >= yb
    prob += lpSum(y_bb) >= int(need_bringback)

    if num_uniques > 0 and history:
        for prev in history:
            forbid = [i for i in idx if df.loc[i, "Name"] in prev and df.loc[i, "PosCanon"] != "G"]
            if forbid:
                limit = max(0, len(forbid) - int(num_uniques))
                prob += lpSum(x[i] for i in forbid) <= limit

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        return (False, [])
    chosen = [i for i in idx if x[i].value() >= 0.99]
    return (True, chosen)


def build_lineups(
    df: pd.DataFrame,
    n: int,
    w_up: float,
    w_con: float,
    w_dud: float,
    min_salary: int,
    max_salary: int,
    max_vs_goalie: int,
    need_ev2: int,
    need_ev3: int,
    need_pp1_2: int,
    need_pp1_3: int,
    need_pp2_2: int,
    need_same2: int,
    need_same3: int,
    need_same4: int,
    need_same5: int,
    need_bringback: int,
    num_uniques: int,
    ceil_mult: float,
    floor_mult: float,
) -> pd.DataFrame:
    players = add_quality_columns(df, ceil_mult=ceil_mult, floor_mult=floor_mult)
    outs = []
    history: List[List[str]] = []
    for k in range(n):
        ok, idxs = solve_single_lineup(
            players,
            w_up,
            w_con,
            w_dud,
            min_salary,
            max_salary,
            max_vs_goalie,
            need_ev2,
            need_ev3,
            need_pp1_2,
            need_pp1_3,
            need_pp2_2,
            need_same2,
            need_same3,
            need_same4,
            need_same5,
            need_bringback,
            num_uniques,
            history if num_uniques > 0 else None,
        )
        if not ok:
            break
        chosen = players.loc[idxs].copy()
        C = chosen[chosen["PosCanon"] == "C"].head(2)
        W = chosen[chosen["PosCanon"] == "W"].head(3)
        D = chosen[chosen["PosCanon"] == "D"].head(2)
        G = chosen[chosen["PosCanon"] == "G"].head(1)
        UTIL = chosen.drop(C.index.union(W.index).union(D.index).union(G.index)).head(1)
        lineup = pd.concat([C, W, D, G, UTIL])
        lineup["LineupID"] = k + 1
        lineup["Slot"] = DK_ROSTER_SLOTS[: len(lineup)]
        outs.append(lineup)
        history.append(lineup[lineup["PosCanon"] != "G"]["Name"].tolist())
    if outs:
        return pd.concat(outs, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

def _load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    log.info("Loading config from %s", path)
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_stack_targets(cfg: Dict[str, object]) -> Dict[str, int]:
    stacks = cfg.get("stacks") or {}
    if not isinstance(stacks, dict):
        return {}
    targets = {"same2": 0, "same3": 0, "same4": 0, "same5": 0, "pp1_2": 0, "pp1_3": 0, "pp2_2": 0}
    min_team = stacks.get("min_team_stacks") or []
    if isinstance(min_team, list):
        for entry in min_team:
            if not isinstance(entry, dict):
                continue
            size = int(entry.get("team_size", 0) or entry.get("min_size", 0))
            if size in (2, 3, 4, 5):
                key = f"same{size}"
                targets[key] = max(targets[key], 1)
    if stacks.get("power_play_correlation"):
        targets["pp1_2"] = max(targets["pp1_2"], 1)
    if stacks.get("power_play_triple"):
        targets["pp1_3"] = max(targets["pp1_3"], 1)
    if stacks.get("pp2_mini"):
        targets["pp2_2"] = max(targets["pp2_2"], 1)
    if stacks.get("avoid_against_goalie"):
        targets["avoid_goalie"] = 1
    return targets


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("NHL Optimizer (NFL-style CLI)")
    ap.add_argument("--labs-dir", type=str, default=DEFAULT_LABS_DIR, help="Directory of FantasyLabs NHL CSVs")
    ap.add_argument("--date", type=str, required=True, help="Slate date (YYYY-MM-DD)")
    ap.add_argument("--out", type=str, default="out/lineups_{date}.csv", help="Output CSV template")
    ap.add_argument("--num-lineups", type=int, default=150, help="Number of lineups to build")
    ap.add_argument("--num-uniques", type=int, default=1, help="Minimum unique players between lineups")
    ap.add_argument("--min-salary", type=int, default=48000, help="Minimum salary for a lineup")
    ap.add_argument("--max-salary", type=int, default=DK_SALARY_CAP, help="Maximum salary for a lineup")
    ap.add_argument("--config", type=str, help="YAML/JSON config for constraints/randomness/stacks")
    ap.add_argument("--c-file", type=str, help="Explicit C projections file")
    ap.add_argument("--w-file", type=str, help="Explicit W projections file")
    ap.add_argument("--d-file", type=str, help="Explicit D projections file")
    ap.add_argument("--g-file", type=str, help="Explicit G projections file")
    ap.add_argument("--export-raw", action="store_true", help="Also write raw (name-only) lineups CSV")
    ap.add_argument("--player-ids", type=str, default=DEFAULT_PLAYER_IDS, help="DraftKings player ID mapping CSV")

    ap.add_argument("--max-vs-goalie", type=int, default=0)
    ap.add_argument("--evline2", type=int, default=0)
    ap.add_argument("--evline3", type=int, default=0)
    ap.add_argument("--pp1_2", type=int, default=0)
    ap.add_argument("--pp1_3", type=int, default=0)
    ap.add_argument("--pp2_2", type=int, default=0)
    ap.add_argument("--same2", type=int, default=0)
    ap.add_argument("--same3", type=int, default=0)
    ap.add_argument("--same4", type=int, default=0)
    ap.add_argument("--same5", type=int, default=0)
    ap.add_argument("--bringback", type=int, default=0)
    ap.add_argument("--w-up", type=float, default=0.15)
    ap.add_argument("--w-con", type=float, default=0.05)
    ap.add_argument("--w-dud", type=float, default=0.03)
    ap.add_argument("--ceil-mult", type=float, default=1.6)
    ap.add_argument("--floor-mult", type=float, default=0.55)
    ap.add_argument("--diversify", type=int, default=1)
    return ap.parse_args(argv)


def _coerce_stack_targets(args: argparse.Namespace, cfg: Dict[str, object]) -> Dict[str, int]:
    targets = {
        "evline2": max(int(args.evline2), 0),
        "evline3": max(int(args.evline3), 0),
        "pp1_2": max(int(args.pp1_2), 0),
        "pp1_3": max(int(args.pp1_3), 0),
        "pp2_2": max(int(args.pp2_2), 0),
        "same2": max(int(args.same2), 0),
        "same3": max(int(args.same3), 0),
        "same4": max(int(args.same4), 0),
        "same5": max(int(args.same5), 0),
        "bringback": max(int(args.bringback), 0),
    }
    derived = _extract_stack_targets(cfg)
    for key, val in derived.items():
        if key in targets and key != "avoid_goalie":
            targets[key] = max(targets[key], int(val))
    if "avoid_goalie" in derived and not cfg.get("max_vs_goalie"):
        targets["avoid_goalie"] = 1
    if any(
        getattr(args, flag)
        for flag in ["evline2", "evline3", "pp1_2", "pp1_3", "pp2_2", "same2", "same3", "same4", "same5", "bringback"]
    ):
        log.warning("Legacy stack flags are deprecated; prefer configuring stacks via --config.")
    return targets


def _determine_num_uniques(args: argparse.Namespace, cfg: Dict[str, object]) -> int:
    if "num_uniques" in cfg:
        return int(cfg["num_uniques"])
    if args.diversify != 1:
        log.warning("--diversify is deprecated; use --num-uniques instead.")
        return int(args.diversify)
    return int(args.num_uniques)


def _resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    overrides = {"C": args.c_file, "W": args.w_file, "D": args.d_file, "G": args.g_file}
    return {k: v for k, v in overrides.items() if v}


def _format_for_export(lineups: pd.DataFrame, pid_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if lineups.empty:
        return (pd.DataFrame(), pd.DataFrame())
    slots = DK_ROSTER_SLOTS
    decorated_rows: List[Dict[str, object]] = []
    raw_rows: List[Dict[str, object]] = []
    grouped = lineups.groupby("LineupID")
    for lid, group in grouped:
        slot_map: Dict[str, Optional[pd.Series]] = {slot: None for slot in slots}
        for _, row in group.iterrows():
            slot = str(row["Slot"])
            if slot in slot_map:
                slot_map[slot] = row
        raw_row: Dict[str, object] = {slot: (slot_map[slot]["Name"] if slot_map[slot] is not None else "") for slot in slots}
        raw_row["TotalSalary"] = int(group["Salary"].sum())
        raw_row["ProjPoints"] = round(float(group.get("ProjMean", group["Proj"]).sum()), 3)
        raw_rows.append(raw_row)

        decorated_row: Dict[str, object] = {}
        for slot in slots:
            rec = slot_map[slot]
            if rec is None:
                decorated_row[slot] = ""
                continue
            decorated_row[slot] = decorate_with_ids(rec["Name"], rec.get("Team"), rec.get("PosCanon"), pid_map)
        decorated_row["TotalSalary"] = raw_row["TotalSalary"]
        decorated_row["ProjPoints"] = raw_row["ProjPoints"]
        decorated_rows.append(decorated_row)
    return (pd.DataFrame(decorated_rows), pd.DataFrame(raw_rows))


def main(argv: Optional[List[str]] = None) -> str:
    args = parse_args(argv)
    cfg = _load_config(args.config)

    labs_dir = args.labs_dir or DEFAULT_LABS_DIR
    overrides = _resolve_paths(args)
    date_token = _sanitize_date_for_loader(args.date)
    raw = load_labs_raw(labs_dir, date_token, overrides if overrides else None)
    pool = _prepare_pool(raw)
    pool = _apply_randomness(pool, cfg)

    num_lineups = int(cfg.get("num_lineups", args.num_lineups))
    num_uniques = _determine_num_uniques(args, cfg)
    min_salary = int(cfg.get("min_salary", args.min_salary))
    max_salary = int(cfg.get("max_salary", args.max_salary))
    stacks = _coerce_stack_targets(args, cfg)
    max_vs_goalie = int(cfg.get("max_vs_goalie", args.max_vs_goalie))
    if stacks.get("avoid_goalie"):
        max_vs_goalie = 0

    weights_cfg = cfg.get("weights") or {}
    w_up = float(weights_cfg.get("upside", args.w_up))
    w_con = float(weights_cfg.get("consistency", args.w_con))
    w_dud = float(weights_cfg.get("duds", args.w_dud))
    ceil_mult = float(cfg.get("ceil_mult", args.ceil_mult))
    floor_mult = float(cfg.get("floor_mult", args.floor_mult))

    lineups = build_lineups(
        pool,
        num_lineups,
        w_up,
        w_con,
        w_dud,
        min_salary,
        max_salary,
        max_vs_goalie,
        stacks.get("evline2", 0),
        stacks.get("evline3", 0),
        stacks.get("pp1_2", 0),
        stacks.get("pp1_3", 0),
        stacks.get("pp2_2", 0),
        stacks.get("same2", 0),
        stacks.get("same3", 0),
        stacks.get("same4", 0),
        stacks.get("same5", 0),
        stacks.get("bringback", 0),
        num_uniques,
        ceil_mult,
        floor_mult,
    )

    if lineups.empty:
        log.error("No lineups produced (solver infeasible — check salary bounds, uniqueness, or stack rules).")
        sys.exit(2)

    pid_map = {}
    player_ids_path = args.player_ids or DEFAULT_PLAYER_IDS
    if os.path.exists(player_ids_path):
        pid_map = load_player_ids(player_ids_path)
    else:
        log.warning("Player ID mapping not found at %s; exporting without IDs.", player_ids_path)

    decorated, raw_df = _format_for_export(lineups, pid_map)
    out_path = (args.out or "out/lineups_{date}.csv").format(date=args.date)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    decorated.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    export_raw = bool(cfg.get("export_raw", args.export_raw))
    if export_raw:
        raw_path = out_path.replace(".csv", "_raw.csv")
        raw_df.to_csv(raw_path, index=False)
        log.info("Wrote %s", raw_path)

    if pid_map:
        missing = decorated.applymap(lambda s: str(s).endswith("()"))
        if missing.any().any():
            log.warning("Some players missing DraftKings IDs; exported as 'Name ()'. Check player_ids.csv")
    return out_path


if __name__ == "__main__":
    main()
