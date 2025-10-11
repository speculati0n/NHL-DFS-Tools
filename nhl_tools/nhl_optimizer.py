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

from .nhl_player_ids import decorate_with_ids, load_player_ids
from .nhl_projection_loader import load_labs_for_date
from .nhl_stacks import game_pairs, line_bucket


log = logging.getLogger("nhl_opt")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DK_SALARY_CAP = 50000
DK_ROSTER_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
DK_ROSTER_COUNTS = {"C": 2, "W": 3, "D": 2, "G": 1}
SKATER_POS = {"C", "W", "D"}
DEFAULT_RANDOMNESS = {"C": 8.0, "W": 12.0, "D": 10.0, "G": 0.0}


def _parse_full(val: object) -> Tuple[Optional[str], Optional[int]]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (None, None)
    text = str(val).strip().upper()
    if not text:
        return (None, None)
    typ = None
    if "D" in text:
        typ = "D"
    elif "F" in text:
        typ = "F"
    line = None
    for token in text.replace("F", " ").replace("D", " ").split():
        if token.isdigit():
            try:
                line = int(token)
                break
            except ValueError:
                continue
    return (typ, line)


def _pp_to_int(val: object) -> Optional[int]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    text = str(val).strip().upper()
    for token in text.replace("PP", " ").split():
        if token.isdigit():
            try:
                return int(token)
            except ValueError:
                continue
    return None


def _clean_team(val: object) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    team = str(val).strip().upper()
    return team or None


def prepare_player_pool(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["Name"] = df["name"].astype(str).str.strip()
    df["Team"] = df["team"].astype(str).str.upper().str.strip()
    df["Opp"] = df.get("opp", np.nan)
    if "Opp" in df.columns:
        df["Opp"] = df["Opp"].astype(str).str.upper().str.strip()
    df["Salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0).astype(int)
    df["Proj"] = pd.to_numeric(df.get("proj_points", 0.0), errors="coerce").fillna(0.0)
    df["ProjBase"] = df["Proj"].astype(float)
    df["Own"] = pd.to_numeric(df.get("projected_own", 0.0), errors="coerce")
    df["FullRaw"] = df.get("full_stack_flag")
    df["PP"] = df.get("powerplay_flag")
    df["PosCanon"] = df.get("pos", "").astype(str).str.upper().str.strip()

    # Derive EV / PP helpers
    parsed = df["FullRaw"].map(_parse_full)
    df["EV_Type"] = parsed.map(lambda x: x[0])
    df["EV_Line"] = parsed.map(lambda x: x[1])
    df["PP_Unit"] = df["PP"].map(_pp_to_int)

    df = df.dropna(subset=["Team", "Salary", "Proj", "PosCanon"])
    df = df.drop_duplicates(subset=["Name", "Team", "PosCanon"], keep="first").reset_index(drop=True)

    def _game_key(row: pd.Series) -> Optional[str]:
        team = _clean_team(row.get("Team"))
        opp = _clean_team(row.get("Opp"))
        if team and opp and team != opp:
            return "@".join(sorted([team, opp]))
        return None

    df["GameKey"] = df.apply(_game_key, axis=1)
    df["IsSkater"] = df["PosCanon"].isin(SKATER_POS)
    df["IsGoalie"] = df["PosCanon"] == "G"

    df = line_bucket(df)
    return df


def apply_randomness(df: pd.DataFrame, randomness_cfg: Optional[Dict[str, float]]) -> pd.DataFrame:
    rp = DEFAULT_RANDOMNESS.copy()
    if randomness_cfg:
        for pos, pct in randomness_cfg.items():
            try:
                rp[pos.upper()] = float(pct)
            except (ValueError, TypeError):
                log.warning("Invalid randomness pct for %s -> %s", pos, pct)
    df = df.copy()
    boosts = []
    for _, row in df.iterrows():
        pct = rp.get(str(row["PosCanon"]).upper(), 0.0)
        base = float(row["ProjBase"])
        jitter = random.uniform(-pct, pct) / 100.0 if pct else 0.0
        boosted = base * (1.0 + jitter)
        boosts.append(boosted)
    df["Proj"] = boosts
    return df


def add_quality_columns(df: pd.DataFrame, ceil_mult: float = 1.6, floor_mult: float = 0.55) -> pd.DataFrame:
    out = df.copy()
    out["Ceil"] = out["Proj"] * float(ceil_mult)
    out["Floor"] = out["Proj"] * float(floor_mult)

    def zcol(col: str, by: str = "PosCanon") -> pd.Series:
        g = out.groupby(by)[col]
        return (out[col] - g.transform("mean")) / (g.transform("std").replace(0, np.nan))

    out["UpsideZ"] = zcol("Ceil")
    out["ConsistencyZ"] = -zcol("Floor")
    out["DudPenalty"] = (3.0 / out["Proj"].clip(lower=1e-3)).clip(upper=2.0)
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
    previous: Optional[List[List[str]]] = None,
    max_team_skaters: Optional[int] = None,
) -> Tuple[bool, List[int]]:
    df = players.reset_index(drop=True).copy()
    N = len(df)
    idx = range(N)
    C = [i for i in idx if df.loc[i, "PosCanon"] == "C"]
    W = [i for i in idx if df.loc[i, "PosCanon"] == "W"]
    D = [i for i in idx if df.loc[i, "PosCanon"] == "D"]
    G = [i for i in idx if df.loc[i, "PosCanon"] == "G"]

    total_players = sum(DK_ROSTER_COUNTS.values()) + 1  # add UTIL

    x = LpVariable.dicts("x", idx, lowBound=0, upBound=1, cat=LpBinary)
    prob = LpProblem("DK_NHL", LpMaximize)

    score = (
        df["Proj"]
        * (1.0 + w_up * df["UpsideZ"].fillna(0.0))
        * (1.0 + w_con * df["ConsistencyZ"].fillna(0.0))
        - w_dud * df["DudPenalty"].fillna(0.0)
    )
    prob += lpSum(x[i] * float(score.loc[i]) for i in idx)

    prob += lpSum(x[i] * int(df.loc[i, "Salary"]) for i in idx) <= int(max_salary)
    prob += lpSum(x[i] * int(df.loc[i, "Salary"]) for i in idx) >= int(min_salary)

    prob += lpSum(x[i] for i in C) >= DK_ROSTER_COUNTS["C"]
    prob += lpSum(x[i] for i in W) >= DK_ROSTER_COUNTS["W"]
    prob += lpSum(x[i] for i in D) >= DK_ROSTER_COUNTS["D"]
    prob += lpSum(x[i] for i in G) == DK_ROSTER_COUNTS["G"]
    prob += lpSum(x[i] for i in idx) == total_players

    max_team = max_team_skaters if max_team_skaters is not None else 5
    teams = df.loc[:, "Team"].astype(str).str.upper()
    for team in teams.unique():
        if not team or team == "NAN":
            continue
        Ti = [i for i in idx if df.loc[i, "Team"] == team and df.loc[i, "PosCanon"] != "G"]
        if Ti:
            prob += lpSum(x[i] for i in Ti) <= max_team

    for gi in G:
        gteam = df.loc[gi, "Team"]
        opp = df.loc[gi, "Opp"]
        if not isinstance(opp, str) or not opp:
            continue
        vs = [i for i in idx if df.loc[i, "Team"] == opp and df.loc[i, "PosCanon"] in SKATER_POS]
        if vs:
            prob += lpSum(x[i] for i in vs) <= int(max_vs_goalie) + (1 - x[gi]) * total_players

    grouped = line_bucket(df)
    groups = {}
    for i, r in grouped.iterrows():
        ev = r.get("EV_LINE_TAG")
        pp = r.get("PP_TAG")
        if ev:
            groups.setdefault(("EV", ev), []).append(i)
        if pp:
            groups.setdefault(("PP", pp), []).append(i)

    ev_keys = [tag for kind, tag in groups.keys() if kind == "EV"]
    y_ev2 = [LpVariable(f"y_ev2_{tag}", 0, 1, LpBinary) for tag in ev_keys]
    for y, tag in zip(y_ev2, ev_keys):
        members = groups[("EV", tag)]
        prob += lpSum(x[i] for i in members) >= 2 * y
    if y_ev2:
        prob += lpSum(y_ev2) >= int(need_ev2)

    y_ev3 = [LpVariable(f"y_ev3_{tag}", 0, 1, LpBinary) for tag in ev_keys]
    for y, tag in zip(y_ev3, ev_keys):
        members = groups[("EV", tag)]
        prob += lpSum(x[i] for i in members) >= 3 * y
    if y_ev3:
        prob += lpSum(y_ev3) >= int(need_ev3)

    pp1_keys = [tag for kind, tag in groups.keys() if kind == "PP" and str(tag).upper().endswith("PP1")]
    y_pp1_2 = [LpVariable(f"y_pp1_2_{tag}", 0, 1, LpBinary) for tag in pp1_keys]
    y_pp1_3 = [LpVariable(f"y_pp1_3_{tag}", 0, 1, LpBinary) for tag in pp1_keys]
    for y2, y3, tag in zip(y_pp1_2, y_pp1_3, pp1_keys):
        mem = groups[("PP", tag)]
        prob += lpSum(x[i] for i in mem) >= 2 * y2
        prob += lpSum(x[i] for i in mem) >= 3 * y3
    if y_pp1_2:
        prob += lpSum(y_pp1_2) >= int(need_pp1_2)
    if y_pp1_3:
        prob += lpSum(y_pp1_3) >= int(need_pp1_3)

    pp2_keys = [tag for kind, tag in groups.keys() if kind == "PP" and str(tag).upper().endswith("PP2")]
    y_pp2_2 = [LpVariable(f"y_pp2_2_{tag}", 0, 1, LpBinary) for tag in pp2_keys]
    for y, tag in zip(y_pp2_2, pp2_keys):
        mem = groups[("PP", tag)]
        prob += lpSum(x[i] for i in mem) >= 2 * y
    if y_pp2_2:
        prob += lpSum(y_pp2_2) >= int(need_pp2_2)

    teams_list = sorted(df["Team"].dropna().unique().tolist())
    y_same2 = [LpVariable(f"y_same2_{t}", 0, 1, LpBinary) for t in teams_list]
    y_same3 = [LpVariable(f"y_same3_{t}", 0, 1, LpBinary) for t in teams_list]
    y_same4 = [LpVariable(f"y_same4_{t}", 0, 1, LpBinary) for t in teams_list]
    y_same5 = [LpVariable(f"y_same5_{t}", 0, 1, LpBinary) for t in teams_list]
    for t, y2, y3, y4, y5 in zip(teams_list, y_same2, y_same3, y_same4, y_same5):
        Ti = [i for i in idx if df.loc[i, "Team"] == t and df.loc[i, "PosCanon"] != "G"]
        if not Ti:
            continue
        prob += lpSum(x[i] for i in Ti) >= 2 * y2
        prob += lpSum(x[i] for i in Ti) >= 3 * y3
        prob += lpSum(x[i] for i in Ti) >= 4 * y4
        prob += lpSum(x[i] for i in Ti) >= 5 * y5
    if y_same2:
        prob += lpSum(y_same2) >= int(need_same2)
    if y_same3:
        prob += lpSum(y_same3) >= int(need_same3)
    if y_same4:
        prob += lpSum(y_same4) >= int(need_same4)
    if y_same5:
        prob += lpSum(y_same5) >= int(need_same5)

    games = game_pairs(df)
    y_bb = [LpVariable(f"y_bb_{a}_{b}", 0, 1, LpBinary) for a, b in games]
    y_stack_a = [LpVariable(f"y_stack_{a}_{b}_{a}", 0, 1, LpBinary) for a, b in games]
    y_stack_b = [LpVariable(f"y_stack_{a}_{b}_{b}", 0, 1, LpBinary) for a, b in games]
    for (a, b), y_game, ya, yb in zip(games, y_bb, y_stack_a, y_stack_b):
        A = [i for i in idx if df.loc[i, "Team"] == a and df.loc[i, "PosCanon"] in SKATER_POS]
        B = [i for i in idx if df.loc[i, "Team"] == b and df.loc[i, "PosCanon"] in SKATER_POS]
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
    if y_bb:
        prob += lpSum(y_bb) >= int(need_bringback)

    if num_uniques > 0 and previous:
        for prior in previous:
            names = set(prior)
            prev_idx = [i for i in idx if df.loc[i, "Name"] in names]
            if prev_idx:
                prob += lpSum(x[i] for i in prev_idx) <= total_players - int(num_uniques)

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        return (False, [])
    chosen = [i for i in idx if x[i].value() and x[i].value() >= 0.99]
    return (True, chosen)


def assign_slots(chosen: pd.DataFrame) -> pd.DataFrame:
    working = chosen.copy()
    working = working.sort_values("Proj", ascending=False)

    def take(pos: str, count: int) -> pd.DataFrame:
        avail = working[working["PosCanon"] == pos]
        picked = avail.head(count)
        return picked

    C = take("C", 2)
    W = take("W", 3)
    D = take("D", 2)
    G = take("G", 1)
    used_idx = set(C.index) | set(W.index) | set(D.index) | set(G.index)
    UTIL = working.drop(index=list(used_idx))
    UTIL = UTIL[UTIL["PosCanon"].isin(SKATER_POS)].head(1)

    lineup = []
    slot_map = {
        "C": ["C1", "C2"],
        "W": ["W1", "W2", "W3"],
        "D": ["D1", "D2"],
        "G": ["G"],
    }
    for pos, slots in slot_map.items():
        frame = {"C": C, "W": W, "D": D, "G": G}[pos]
        for slot, (_, row) in zip(slots, frame.iterrows()):
            entry = row.to_dict()
            entry["Slot"] = slot
            lineup.append(entry)
    if not UTIL.empty:
        urow = UTIL.iloc[0].to_dict()
        urow["Slot"] = "UTIL"
        lineup.append(urow)
    else:
        leftovers = working.drop(index=list(used_idx))
        if not leftovers.empty:
            urow = leftovers.iloc[0].to_dict()
            urow["Slot"] = "UTIL"
            lineup.append(urow)

    return pd.DataFrame(lineup)


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
    max_team_skaters: Optional[int],
) -> pd.DataFrame:
    players = df.copy()
    outs: List[pd.DataFrame] = []
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
            previous=history,
            max_team_skaters=max_team_skaters,
        )
        if not ok:
            break
        chosen = players.loc[idxs].copy()
        lineup = assign_slots(chosen)
        lineup["LineupID"] = k + 1
        outs.append(lineup)
        history.append(lineup["Name"].tolist())
    if outs:
        return pd.concat(outs, ignore_index=True)
    return pd.DataFrame()


def format_exports(lineups: pd.DataFrame, pid_map: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    raw_rows = []
    missing: set[str] = set()
    for lid in sorted(lineups["LineupID"].unique()):
        subset = lineups[lineups["LineupID"] == lid]
        row: Dict[str, object] = {}
        raw: Dict[str, object] = {}
        row["LineupID"] = lid
        raw["LineupID"] = lid
        for slot in DK_ROSTER_SLOTS:
            player = subset[subset["Slot"] == slot]
            if player.empty:
                row[slot] = ""
                raw[slot] = ""
                continue
            rec = player.iloc[0]
            decorated = decorate_with_ids(rec["Name"], rec.get("Team", ""), rec.get("PosCanon", ""), pid_map)
            if decorated.endswith(" ()"):
                missing.add(f"{rec['Name']}|{rec.get('Team','')}|{rec.get('PosCanon','')}")
            row[slot] = decorated
            raw[slot] = rec["Name"]
        total_salary = int(subset["Salary"].sum())
        total_proj = float(subset.get("ProjBase", subset["Proj"]).sum())
        row["TotalSalary"] = total_salary
        row["ProjPoints"] = round(total_proj, 3)
        raw["TotalSalary"] = total_salary
        raw["ProjPoints"] = round(total_proj, 3)
        rows.append(row)
        raw_rows.append(raw)
    if missing:
        log.warning("Missing player IDs for %d players. They appear as Name ().", len(missing))
    return pd.DataFrame(rows), pd.DataFrame(raw_rows)


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("NHL Optimizer (NFL-style CLI)")
    ap.add_argument("--labs-dir", type=str, default="dk_data", help="Directory of FantasyLabs NHL CSVs")
    ap.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", type=str, default="out/lineups_{date}.csv")
    ap.add_argument("--num-lineups", type=int, default=150)
    ap.add_argument("--num-uniques", type=int, default=1)
    ap.add_argument("--min-salary", type=int, default=48000)
    ap.add_argument("--max-salary", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--config", type=str, help="YAML/JSON config for constraints/randomness/stacks")
    ap.add_argument("--c-file", type=str, help="Explicit path to C projections")
    ap.add_argument("--w-file", type=str, help="Explicit path to W projections")
    ap.add_argument("--d-file", type=str, help="Explicit path to D projections")
    ap.add_argument("--g-file", type=str, help="Explicit path to G projections")
    ap.add_argument("--export-raw", action="store_true", help="Also write a raw file without IDs")
    ap.add_argument("--player-ids", type=str, default="dk_data/player_ids.csv", help="DraftKings player id mapping")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for projection variance")

    # legacy flags (parsed but deprecated)
    ap.add_argument("--max-vs-goalie", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--evline2", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--evline3", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pp1_2", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pp1_3", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--pp2_2", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--same2", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--same3", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--same4", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--same5", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--bringback", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--w-up", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--w-con", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--w-dud", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ceil-mult", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--floor-mult", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--diversify", type=int, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def merge_settings(args: argparse.Namespace, cfg: Dict) -> Dict[str, object]:
    settings: Dict[str, object] = {}
    settings["num_lineups"] = int(cfg.get("num_lineups", args.num_lineups))
    settings["num_uniques"] = int(cfg.get("num_uniques", args.num_uniques))
    settings["min_salary"] = int(cfg.get("min_salary", args.min_salary))
    max_salary = int(cfg.get("max_salary", args.max_salary))
    settings["max_salary"] = min(max_salary, DK_SALARY_CAP)

    stack_cfg = cfg.get("stacks", {}) if isinstance(cfg.get("stacks"), dict) else {}
    randomness_cfg = cfg.get("randomness_pct_by_pos") if isinstance(cfg.get("randomness_pct_by_pos"), dict) else None
    settings["randomness_pct_by_pos"] = randomness_cfg
    settings["seed"] = cfg.get("seed", args.seed)

    def legacy_warn(flag: str):
        if getattr(args, flag.replace("-", "_")) is not None:
            log.warning("Flag --%s is deprecated. Prefer using --config with NFL-style keys.", flag)

    max_vs_goalie = stack_cfg.get("max_vs_goalie")
    if max_vs_goalie is None and stack_cfg.get("avoid_against_goalie"):
        max_vs_goalie = 0
    if max_vs_goalie is None:
        max_vs_goalie = getattr(args, "max_vs_goalie") if args.max_vs_goalie is not None else 0
    else:
        legacy_warn("max-vs-goalie")
    settings["max_vs_goalie"] = int(max_vs_goalie)

    def setting_from(key: str, legacy_flag: str, default: int = 0) -> int:
        if key in cfg:
            return int(cfg[key])
        val = getattr(args, legacy_flag)
        if val is not None:
            legacy_warn(legacy_flag.replace("_", "-"))
            return int(val)
        return default

    settings["need_ev2"] = setting_from("evline2", "evline2")
    settings["need_ev3"] = setting_from("evline3", "evline3")
    settings["need_pp1_2"] = setting_from("pp1_2", "pp1_2")
    settings["need_pp1_3"] = setting_from("pp1_3", "pp1_3")
    settings["need_pp2_2"] = setting_from("pp2_2", "pp2_2")
    settings["need_same2"] = setting_from("same2", "same2")
    settings["need_same3"] = setting_from("same3", "same3")
    settings["need_same4"] = setting_from("same4", "same4")
    settings["need_same5"] = setting_from("same5", "same5")
    settings["need_bringback"] = setting_from("bringback", "bringback")

    team_stacks = stack_cfg.get("min_team_stacks")
    if isinstance(team_stacks, list):
        size_counts: Dict[int, int] = {}
        max_team_cap = None
        for entry in team_stacks:
            if not isinstance(entry, dict):
                continue
            size = int(entry.get("team_size", 0))
            if size <= 0:
                continue
            count = int(entry.get("min", 1))
            size_counts[size] = max(size_counts.get(size, 0), count)
            if entry.get("max_from_team") is not None:
                cap = int(entry["max_from_team"])
                max_team_cap = cap if max_team_cap is None else min(max_team_cap, cap)
        if 2 in size_counts:
            settings["need_same2"] = max(settings["need_same2"], size_counts[2])
        if 3 in size_counts:
            settings["need_same3"] = max(settings["need_same3"], size_counts[3])
        if 4 in size_counts:
            settings["need_same4"] = max(settings["need_same4"], size_counts[4])
        if 5 in size_counts:
            settings["need_same5"] = max(settings["need_same5"], size_counts[5])
        settings["max_team_skaters"] = max_team_cap
    else:
        settings["max_team_skaters"] = None

    if stack_cfg.get("power_play_correlation"):
        settings["need_pp1_2"] = max(settings["need_pp1_2"], 1)

    def weight_from(key: str, legacy_flag: str, default: float) -> float:
        if key in cfg:
            return float(cfg[key])
        val = getattr(args, legacy_flag)
        if val is not None:
            legacy_warn(legacy_flag.replace("_", "-"))
            return float(val)
        return default

    settings["w_up"] = weight_from("w_up", "w_up", 0.15)
    settings["w_con"] = weight_from("w_con", "w_con", 0.05)
    settings["w_dud"] = weight_from("w_dud", "w_dud", 0.03)
    settings["ceil_mult"] = weight_from("ceil_mult", "ceil_mult", 1.6)
    settings["floor_mult"] = weight_from("floor_mult", "floor_mult", 0.55)

    if getattr(args, "diversify") is not None:
        legacy_warn("diversify")
        settings["num_uniques"] = max(settings["num_uniques"], int(args.diversify))

    return settings


def main() -> Optional[str]:
    args = parse_args()
    cfg = load_config(args.config)
    settings = merge_settings(args, cfg)

    explicit = {"C": args.c_file, "W": args.w_file, "D": args.d_file, "G": args.g_file}
    projections = load_labs_for_date(args.labs_dir, args.date, explicit)
    if settings.get("seed") is not None:
        try:
            random.seed(int(settings["seed"]))
        except Exception:
            log.warning("Invalid seed value %s", settings["seed"])
    pool = prepare_player_pool(projections)
    pool = apply_randomness(pool, settings.get("randomness_pct_by_pos"))
    pool = add_quality_columns(pool, settings["ceil_mult"], settings["floor_mult"])

    lineups = build_lineups(
        pool,
        settings["num_lineups"],
        settings["w_up"],
        settings["w_con"],
        settings["w_dud"],
        settings["min_salary"],
        settings["max_salary"],
        settings["max_vs_goalie"],
        settings["need_ev2"],
        settings["need_ev3"],
        settings["need_pp1_2"],
        settings["need_pp1_3"],
        settings["need_pp2_2"],
        settings["need_same2"],
        settings["need_same3"],
        settings["need_same4"],
        settings["need_same5"],
        settings["need_bringback"],
        settings["num_uniques"],
        settings.get("max_team_skaters"),
    )

    if lineups.empty:
        log.error(
            "No lineups produced (solver infeasible — check salary bounds, uniques, exposures, or pool size)."
        )
        sys.exit(2)

    pid_map = load_player_ids(args.player_ids)
    out_path = (args.out or "out/lineups_{date}.csv").format(date=args.date)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    decorated, raw = format_exports(lineups, pid_map)
    decorated.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)

    if args.export_raw or cfg.get("export_raw"):
        raw_path = out_path.replace(".csv", "_raw.csv")
        raw_dir = os.path.dirname(raw_path) or "."
        os.makedirs(raw_dir, exist_ok=True)
        raw.to_csv(raw_path, index=False)
        log.info("Wrote %s", raw_path)

    return out_path


if __name__ == "__main__":
    main()

