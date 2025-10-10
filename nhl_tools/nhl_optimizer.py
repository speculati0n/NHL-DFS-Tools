from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pulp import (PULP_CBC_CMD, LpBinary, LpMaximize, LpProblem, LpStatusOptimal,
                  LpVariable, lpSum)

from .nhl_data import DK_ROSTER, DK_SALARY_CAP, load_labs_for_date
from .nhl_stacks import game_pairs, line_bucket


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LABS_DIR = os.path.join(REPO_ROOT, "dk_data")
TOTAL_ROSTER_SLOTS = sum(DK_ROSTER.values())


def _print_constraint_summary(*, n, min_salary, max_vs_goalie, team_cap, stack_requirements,
                              bringback_min, min_uniques, has_own, ownership_enabled,
                              total_own_max, chalk_thresh, max_chalk) -> None:
    print(
        "[NHL][Constraints] "
        f"n={n}, min_salary={min_salary}, max_vs_goalie={max_vs_goalie}, team_cap={team_cap}; "
        f"stacks={stack_requirements}, bringback_min={bringback_min}, min_uniques={min_uniques}; "
        "ownership("
        f"has={has_own}, enabled={ownership_enabled}, total_max={total_own_max}, "
        f"chalk_thresh={chalk_thresh}, max_chalk={max_chalk})"
    )


def _cfg_get(cfg: Dict[str, Any], path: Sequence[str], default: Any) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def add_quality_columns(
    df: pd.DataFrame,
    ceil_mult: float = 1.6,
    floor_mult: float = 0.55,
) -> pd.DataFrame:
    """Add projection quality columns used by the objective."""

    out = df.copy()
    out["Ceil"] = out["Proj"] * float(ceil_mult)
    out["Floor"] = out["Proj"] * float(floor_mult)

    def z_score(col: str, by: str = "PosCanon") -> pd.Series:
        grp = out.groupby(by)[col]
        std = grp.transform("std").replace(0, np.nan)
        return (out[col] - grp.transform("mean")) / std

    out["UpsideZ"] = z_score("Ceil").fillna(0.0)
    out["ConsistencyZ"] = -z_score("Floor").fillna(0.0)
    out["DudPenalty"] = (3.0 / out["Proj"].clip(lower=1e-3)).clip(upper=2.0)
    return out


def _prepare_player_pool(
    df: pd.DataFrame,
    ceil_mult: float,
    floor_mult: float,
    cfg: Dict[str, Any],
) -> tuple[pd.DataFrame, bool, bool, float, float, int]:
    """Return enriched projection frame and ownership flags."""

    players = add_quality_columns(df, ceil_mult=ceil_mult, floor_mult=floor_mult)
    players = players.reset_index(drop=True)
    players = line_bucket(players)
    has_own = False
    ownership_enabled = bool(_cfg_get(cfg, ["ownership", "enabled"], True))
    chalk_thresh = float(_cfg_get(cfg, ["ownership", "chalk_thresh"], 20.0))
    max_chalk = int(_cfg_get(cfg, ["ownership", "max_chalk"], 2))

    if "Own" in players.columns:
        own = pd.to_numeric(players["Own"], errors="coerce")
        players["Own"] = own.fillna(0.0)
        if own.notna().mean() > 0.2:
            has_own = True
            eps = 1e-3
            proj_per_own = players["Proj"] / np.maximum(players["Own"].clip(lower=eps), eps)
            mean = proj_per_own.groupby(players["PosCanon"]).transform("mean")
            std = proj_per_own.groupby(players["PosCanon"]).transform("std").replace(0, np.nan)
            ppo_z = (proj_per_own - mean) / std
            players["PPO_Z"] = ppo_z.fillna(0.0)
        else:
            players["PPO_Z"] = 0.0
    else:
        players["Own"] = 0.0
        players["PPO_Z"] = 0.0

    if not has_own:
        fallback_max_chalk = int(_cfg_get(cfg, ["fallback_if_no_own", "max_chalk_when_no_own"], 9))
        max_chalk = fallback_max_chalk
        if _cfg_get(cfg, ["fallback_if_no_own", "disable_ownership_terms"], True):
            ownership_enabled = False
            players["PPO_Z"] = 0.0

    total_own_max = float(_cfg_get(cfg, ["ownership", "total_own_max"], 999.0))
    return players, has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk


def _salary_leftover_choice(cfg: Dict[str, Any], rng: np.random.Generator) -> int:
    mode = str(_cfg_get(cfg, ["random", "leftover_mode"], "mix"))
    buckets = _cfg_get(cfg, ["random", "leftover_buckets", mode], None)
    if not buckets:
        buckets = [0]
    return int(rng.choice(list(buckets)))


def _jitter_weight(base: float, pct: float, rng: np.random.Generator) -> float:
    if base == 0:
        return 0.0
    pct = max(float(pct), 0.0)
    low = max(0.0, 1.0 - pct)
    high = 1.0 + pct
    return float(base) * float(rng.uniform(low, high))


def _summarise_lineup(
    players: pd.DataFrame,
    idxs: Sequence[int],
    has_own: bool,
    chalk_thresh: float,
    extras: Dict[str, Any],
    prior_lineups: List[List[int]],
) -> str:
    chosen = players.loc[list(idxs)].copy()
    salary = int(chosen["Salary"].sum())
    leftover = DK_SALARY_CAP - salary
    own_sum = float(chosen["Own"].sum()) if has_own else None
    chalk_cnt = 0
    if has_own:
        chalk_cnt = int((chosen["Own"] >= chalk_thresh).sum())
    uniques_vs_prior = TOTAL_ROSTER_SLOTS
    if prior_lineups:
        overlaps = []
        chosen_set = set(idxs)
        for prev in prior_lineups:
            overlap = len(chosen_set.intersection(prev))
            overlaps.append(TOTAL_ROSTER_SLOTS - overlap)
        uniques_vs_prior = min(overlaps) if overlaps else TOTAL_ROSTER_SLOTS

    own_str = f"own {own_sum:.1f}" if own_sum is not None else "own n/a"
    summary = (
        f"salary {salary} (left {leftover}), {own_str}, chalk {chalk_cnt}, "
        f"EV3 {extras.get('ev3', 0)}, PP1 {extras.get('pp1', 0)}, "
        f"bringbacks {extras.get('bringbacks', 0)}, uniques {uniques_vs_prior}"
    )
    return summary


def _finalise_lineup(players: pd.DataFrame, idxs: Sequence[int], lineup_id: int) -> pd.DataFrame:
    chosen = players.loc[list(idxs)].copy()
    C = chosen[chosen["PosCanon"] == "C"].head(2)
    W = chosen[chosen["PosCanon"] == "W"].head(3)
    D = chosen[chosen["PosCanon"] == "D"].head(2)
    G = chosen[chosen["PosCanon"] == "G"].head(1)
    util = chosen.drop(C.index.union(W.index).union(D.index).union(G.index)).head(1)
    lineup = pd.concat([C, W, D, G, util])
    slots = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
    lineup["LineupID"] = lineup_id
    lineup["Slot"] = slots[: len(lineup)]
    return lineup


def solve_single_lineup(
    players: pd.DataFrame,
    *,
    min_salary: int,
    max_vs_goalie: int,
    stack_requirements: Dict[str, int],
    bringback_min: int,
    bringback_max: int,
    min_uniques: int,
    prior_lineups: List[List[int]],
    w_up: float,
    w_con: float,
    w_dud: float,
    w_eff: float,
    w_ev3: float,
    w_pp1: float,
    noise: np.ndarray,
    has_own: bool,
    ownership_enabled: bool,
    total_own_max: float,
    chalk_thresh: float,
    max_chalk: int,
) -> tuple[bool, List[int], Dict[str, Any]]:
    df = players.reset_index(drop=True)
    idx = list(range(len(df)))
    C = [i for i in idx if df.loc[i, "PosCanon"] == "C"]
    W = [i for i in idx if df.loc[i, "PosCanon"] == "W"]
    D = [i for i in idx if df.loc[i, "PosCanon"] == "D"]
    G = [i for i in idx if df.loc[i, "PosCanon"] == "G"]
    SK = [i for i in idx if i not in G]

    prob = LpProblem("DK_NHL", LpMaximize)
    x = {i: LpVariable(f"x_{i}", 0, 1, LpBinary) for i in idx}

    base_score = df["Proj"]
    if w_up:
        base_score = base_score * (1.0 + w_up * df["UpsideZ"].fillna(0.0))
    if w_con:
        base_score = base_score * (1.0 + w_con * df["ConsistencyZ"].fillna(0.0))
    if w_dud:
        base_score = base_score - w_dud * df["DudPenalty"].fillna(0.0)
    if ownership_enabled and w_eff:
        base_score = base_score + w_eff * df["PPO_Z"].fillna(0.0)

    noise_vec = noise if noise is not None else np.zeros(len(df))
    objective = lpSum(x[i] * float(base_score.iloc[i] + noise_vec[i]) for i in idx)

    # Stack activation vars for correlation bonuses
    df_groups = {}
    for i, row in df.iterrows():
        ev_tag = row.get("EV_LINE_TAG")
        pp_tag = row.get("PP_TAG")
        if ev_tag:
            df_groups.setdefault(("EV", str(ev_tag)), []).append(i)
        if pp_tag:
            df_groups.setdefault(("PP", str(pp_tag)), []).append(i)

    ev3_vars: List[LpVariable] = []
    ev2_vars: List[LpVariable] = []
    pp1_3_vars: List[LpVariable] = []
    pp1_2_vars: List[LpVariable] = []
    pp2_2_vars: List[LpVariable] = []

    for (kind, tag), members in df_groups.items():
        if kind == "EV":
            y2 = LpVariable(f"y_ev2_{tag}", 0, 1, LpBinary)
            y3 = LpVariable(f"y_ev3_{tag}", 0, 1, LpBinary)
            count = lpSum(x[i] for i in members)
            prob += count >= 2 * y2
            prob += count <= 1 + (len(members) - 1) * y2
            prob += count >= 3 * y3
            prob += count <= 2 + (len(members) - 2) * y3
            prob += y3 <= y2
            ev2_vars.append(y2)
            ev3_vars.append(y3)
        elif kind == "PP":
            if str(tag).endswith("PP1"):
                y2 = LpVariable(f"y_pp1_2_{tag}", 0, 1, LpBinary)
                y3 = LpVariable(f"y_pp1_3_{tag}", 0, 1, LpBinary)
                count = lpSum(x[i] for i in members)
                prob += count >= 2 * y2
                prob += count <= 1 + (len(members) - 1) * y2
                prob += count >= 3 * y3
                prob += count <= 2 + (len(members) - 2) * y3
                prob += y3 <= y2
                pp1_2_vars.append(y2)
                pp1_3_vars.append(y3)
            elif str(tag).endswith("PP2"):
                y2 = LpVariable(f"y_pp2_2_{tag}", 0, 1, LpBinary)
                count = lpSum(x[i] for i in members)
                prob += count >= 2 * y2
                prob += count <= 1 + (len(members) - 1) * y2
                pp2_2_vars.append(y2)

    if w_ev3:
        objective += w_ev3 * lpSum(ev3_vars)
    if w_pp1:
        objective += w_pp1 * lpSum(pp1_3_vars)

    prob += objective

    # Salary constraints
    prob += lpSum(x[i] * int(df.loc[i, "Salary"]) for i in idx) <= DK_SALARY_CAP
    prob += lpSum(x[i] * int(df.loc[i, "Salary"]) for i in idx) >= int(min_salary)

    # Roster composition
    prob += lpSum(x[i] for i in C) >= DK_ROSTER["C"]
    prob += lpSum(x[i] for i in W) >= DK_ROSTER["W"]
    prob += lpSum(x[i] for i in D) >= DK_ROSTER["D"]
    prob += lpSum(x[i] for i in G) == DK_ROSTER["G"]
    prob += lpSum(x[i] for i in idx) == TOTAL_ROSTER_SLOTS

    for team, _ in df.loc[SK, "Team"].value_counts().items():
        team_idx = [i for i in SK if df.loc[i, "Team"] == team]
        prob += lpSum(x[i] for i in team_idx) <= 5

    for gi in G:
        g_team = df.loc[gi, "Team"]
        opp = df.loc[gi, "Opp"]
        if not isinstance(opp, str):
            continue
        opp_idx = [i for i in SK if df.loc[i, "Team"] == opp]
        prob += lpSum(x[i] for i in opp_idx) <= max_vs_goalie + (1 - x[gi]) * TOTAL_ROSTER_SLOTS

    prob += lpSum(ev2_vars) >= int(stack_requirements.get("evline2", 0))
    prob += lpSum(ev3_vars) >= int(stack_requirements.get("evline3", 0))
    prob += lpSum(pp1_2_vars) >= int(stack_requirements.get("pp1_2", 0))
    prob += lpSum(pp1_3_vars) >= int(stack_requirements.get("pp1_3", 0))
    prob += lpSum(pp2_2_vars) >= int(stack_requirements.get("pp2_2", 0))

    teams = sorted(df["Team"].dropna().unique().tolist())
    same_counts = {2: [], 3: [], 4: [], 5: []}
    for team in teams:
        members = [i for i in SK if df.loc[i, "Team"] == team]
        if not members:
            continue
        for size in (2, 3, 4, 5):
            y = LpVariable(f"y_same{size}_{team}", 0, 1, LpBinary)
            prob += lpSum(x[i] for i in members) >= size * y
            prob += lpSum(x[i] for i in members) <= len(members) * y
            same_counts[size].append(y)
    prob += lpSum(same_counts[2]) >= int(stack_requirements.get("same2", 0))
    prob += lpSum(same_counts[3]) >= int(stack_requirements.get("same3", 0))
    prob += lpSum(same_counts[4]) >= int(stack_requirements.get("same4", 0))
    prob += lpSum(same_counts[5]) >= int(stack_requirements.get("same5", 0))

    games = game_pairs(df)
    bringback_vars: List[LpVariable] = []
    for a, b in games:
        A = [i for i in SK if str(df.loc[i, "Team"]).upper() == a]
        B = [i for i in SK if str(df.loc[i, "Team"]).upper() == b]
        if not A or not B:
            continue
        y = LpVariable(f"y_bringback_{a}_{b}", 0, 1, LpBinary)
        stack_a = LpVariable(f"y_stack_{a}_{b}_{a}", 0, 1, LpBinary)
        stack_b = LpVariable(f"y_stack_{a}_{b}_{b}", 0, 1, LpBinary)
        prob += lpSum(x[i] for i in A) >= 2 * stack_a
        prob += lpSum(x[i] for i in B) >= stack_a
        prob += lpSum(x[i] for i in B) >= 2 * stack_b
        prob += lpSum(x[i] for i in A) >= stack_b
        prob += y >= stack_a
        prob += y >= stack_b
        prob += y <= stack_a + stack_b
        bringback_vars.append(y)
    prob += lpSum(bringback_vars) >= int(bringback_min)
    if bringback_max >= 0:
        prob += lpSum(bringback_vars) <= int(bringback_max)

    if ownership_enabled and has_own:
        own_vals = df["Own"].fillna(0.0)
        prob += lpSum(x[i] * float(own_vals.iloc[i]) for i in idx) <= float(total_own_max)
        chalk_idx = [i for i in idx if float(own_vals.iloc[i]) >= chalk_thresh]
        if chalk_idx:
            prob += lpSum(x[i] for i in chalk_idx) <= int(max_chalk)

    if min_uniques > 0 and prior_lineups:
        for prev in prior_lineups:
            overlap_set = [i for i in prev if i < len(df)]
            if not overlap_set:
                continue
            prob += lpSum(x[i] for i in overlap_set) <= TOTAL_ROSTER_SLOTS - min_uniques

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        return False, [], {}

    chosen = [i for i in idx if x[i].value() and x[i].value() >= 0.99]
    extras = {
        "ev3": int(sum(1 for var in ev3_vars if var.value() and var.value() >= 0.5)),
        "pp1": int(sum(1 for var in pp1_3_vars if var.value() and var.value() >= 0.5)),
        "bringbacks": int(sum(1 for var in bringback_vars if var.value() and var.value() >= 0.5)),
    }
    return True, chosen, extras


def build_lineups(
    players: pd.DataFrame,
    *,
    cfg: Dict[str, Any],
    n: int,
    rng: np.random.Generator,
    min_salary_floor: int,
    max_vs_goalie: int,
    stack_requirements: Dict[str, int],
    bringback_min: int,
    w_up: float,
    w_con: float,
    w_dud: float,
    ownership_ctx: Tuple[bool, bool, float, float, int],
    min_uniques: int,
) -> pd.DataFrame:
    has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk = ownership_ctx
    weight_jitter_pct = float(_cfg_get(cfg, ["random", "weight_jitter_pct"], 0.0))
    objective_noise_std = float(_cfg_get(cfg, ["random", "objective_noise_std"], 0.0))
    w_eff_base = float(_cfg_get(cfg, ["ownership", "w_eff"], 0.0)) if ownership_enabled and has_own else 0.0
    w_ev_base = float(_cfg_get(cfg, ["correlation", "w_ev_stack_3p"], 0.0))
    w_pp1_base = float(_cfg_get(cfg, ["correlation", "w_pp1_stack_3p"], 0.0))
    bringback_max = int(_cfg_get(cfg, ["correlation", "bringbacks_max"], 999))

    lineups: List[pd.DataFrame] = []
    prior_lineups: List[List[int]] = []

    print(f"Has ownership data: {has_own} (terms enabled: {ownership_enabled})")

    for k in range(n):
        leftover = _salary_leftover_choice(cfg, rng)
        enforce_min = bool(_cfg_get(cfg, ["salary", "enforce_min"], True))
        min_salary = DK_SALARY_CAP - leftover if enforce_min else min_salary_floor
        min_salary = max(min_salary, min_salary_floor)

        w_eff = _jitter_weight(w_eff_base, weight_jitter_pct, rng)
        w_ev = _jitter_weight(w_ev_base, weight_jitter_pct, rng)
        w_pp1 = _jitter_weight(w_pp1_base, weight_jitter_pct, rng)
        noise = rng.normal(0.0, objective_noise_std, size=len(players)) if objective_noise_std > 0 else np.zeros(len(players))

        # Try multiple leftover targets before giving up on lineup k
        buckets_cfg = _cfg_get(cfg, ["random", "leftover_buckets"], {"mix": [0, 100, 200]})
        mode = str(_cfg_get(cfg, ["random", "leftover_mode"], "mix"))
        bucket_list = list(buckets_cfg.get(mode, [0, 100, 200]))
        # Heuristic: try non-zero first, then zero last (if present)
        bucket_list_sorted = sorted(set(bucket_list), key=lambda x: (x == 0, x))

        ok = False
        idxs: List[int] = []
        extras: Dict[str, Any] = {}
        tried: List[int] = []

        for lb in bucket_list_sorted:
            tried.append(lb)
            _min_salary = DK_SALARY_CAP - int(lb) if enforce_min else min_salary_floor
            _min_salary = max(_min_salary, min_salary_floor)

            ok, idxs, extras = solve_single_lineup(
                players,
                min_salary=int(_min_salary),
                max_vs_goalie=max_vs_goalie,
                stack_requirements=stack_requirements,
                bringback_min=bringback_min,
                bringback_max=bringback_max,
                min_uniques=min_uniques,
                prior_lineups=prior_lineups,
                w_up=w_up,
                w_con=w_con,
                w_dud=w_dud,
                w_eff=w_eff,
                w_ev3=w_ev,
                w_pp1=w_pp1,
                noise=noise,
                has_own=has_own,
                ownership_enabled=ownership_enabled,
                total_own_max=total_own_max,
                chalk_thresh=chalk_thresh,
                max_chalk=max_chalk,
            )
            if ok:
                break

        if not ok:
            print(
                f"Lineup {k}: infeasible for leftover buckets {tried} — continuing to fallback/next lineup."
            )
            continue

        summary = _summarise_lineup(players, idxs, has_own, chalk_thresh, extras, prior_lineups)
        print(f"[NHL] Built lineup {k + 1}: {summary}")

        lineup = _finalise_lineup(players, idxs, k + 1)
        lineups.append(lineup)
        prior_lineups.append(list(idxs))

    if lineups:
        return pd.concat(lineups, ignore_index=True)
    return pd.DataFrame()


def _try_build(
    players,
    *,
    cfg,
    num_lineups,
    min_salary,
    max_vs_goalie,
    stack_requirements,
    bringback,
    w_up,
    w_con,
    w_dud,
    ownership_ctx,
    min_uniques,
    rng,
):
    return build_lineups(
        players,
        cfg=cfg,
        n=num_lineups,
        rng=rng,
        min_salary_floor=min_salary,
        max_vs_goalie=max_vs_goalie,
        stack_requirements=stack_requirements,
        bringback_min=bringback,
        w_up=w_up,
        w_con=w_con,
        w_dud=w_dud,
        ownership_ctx=ownership_ctx,
        min_uniques=min_uniques,
    )


def main(
    *,
    cfg: Dict[str, Any],
    labs_dir: str = DEFAULT_LABS_DIR,
    date: str,
    out: str,
    num_lineups: int = 20,
    min_salary: int = 49500,
    max_vs_goalie: int = 0,
    evline2: int = 0,
    evline3: int = 0,
    pp1_2: int = 0,
    pp1_3: int = 0,
    pp2_2: int = 0,
    same2: int = 0,
    same3: int = 0,
    same4: int = 0,
    same5: int = 0,
    bringback: int = 0,
    w_up: float = 0.15,
    w_con: float = 0.05,
    w_dud: float = 0.03,
    ceil_mult: float = 1.6,
    floor_mult: float = 0.55,
    diversify: int = 1,
) -> None:
    if cfg is None:
        raise ValueError("cfg must be provided")

    labs_dir = labs_dir or DEFAULT_LABS_DIR
    df = load_labs_for_date(labs_dir, date)
    players, has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk = _prepare_player_pool(
        df, ceil_mult=ceil_mult, floor_mult=floor_mult, cfg=cfg
    )

    min_uniques_cfg = int(_cfg_get(cfg, ["uniqueness", "min_uniques"], 1))
    min_uniques = max(min_uniques_cfg, int(diversify))
    min_uniques = min(min_uniques, TOTAL_ROSTER_SLOTS)

    seed = int(_cfg_get(cfg, ["random", "seed"], 42))
    rng = np.random.default_rng(seed)

    stack_requirements = {
        "evline2": evline2,
        "evline3": evline3,
        "pp1_2": pp1_2,
        "pp1_3": pp1_3,
        "pp2_2": pp2_2,
        "same2": same2,
        "same3": same3,
        "same4": same4,
        "same5": same5,
    }

    # Incompatibility guard: bringback requires allowing at least one skater vs G
    if bringback > 0 and max_vs_goalie == 0:
        print("[NHL][Adjust] bringback > 0 but max_vs_goalie == 0 → setting max_vs_goalie = 1")
        max_vs_goalie = 1

    _print_constraint_summary(
        n=num_lineups,
        min_salary=min_salary,
        max_vs_goalie=max_vs_goalie,
        team_cap=5,
        stack_requirements=stack_requirements,
        bringback_min=bringback,
        min_uniques=min_uniques,
        has_own=has_own,
        ownership_enabled=ownership_enabled,
        total_own_max=total_own_max,
        chalk_thresh=chalk_thresh,
        max_chalk=max_chalk,
    )

    lineup_df = _try_build(
        players,
        cfg=cfg,
        num_lineups=num_lineups,
        min_salary=min_salary,
        max_vs_goalie=max_vs_goalie,
        stack_requirements=stack_requirements,
        bringback=bringback,
        w_up=w_up,
        w_con=w_con,
        w_dud=w_dud,
        ownership_ctx=(has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk),
        min_uniques=min_uniques,
        rng=rng,
    )

    new_max_vs_goalie = max_vs_goalie
    min_salary_floor = min_salary
    min_uniques2 = min_uniques

    if lineup_df.empty:
        print("[NHL][AutoRelax] No lineups on first attempt — applying fallback ladder.")
        # Step A: allow at least one skater vs goalie
        new_max_vs_goalie = max(1, max_vs_goalie)
        if new_max_vs_goalie != max_vs_goalie:
            print(f"[NHL][AutoRelax] max_vs_goalie: {max_vs_goalie} → {new_max_vs_goalie}")
        # Step B: relax min_salary in 500 steps down to a sensible floor
        min_salary_floor = max(48000, min_salary - 500)
        if min_salary_floor != min_salary:
            print(f"[NHL][AutoRelax] min_salary: {min_salary} → {min_salary_floor}")
        # Try again
        lineup_df = _try_build(
            players,
            cfg=cfg,
            num_lineups=num_lineups,
            min_salary=min_salary_floor,
            max_vs_goalie=new_max_vs_goalie,
            stack_requirements=stack_requirements,
            bringback=bringback,
            w_up=w_up,
            w_con=w_con,
            w_dud=w_dud,
            ownership_ctx=(has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk),
            min_uniques=min_uniques,
            rng=rng,
        )

    if lineup_df.empty:
        # Step C: uniqueness → 1
        if min_uniques > 1:
            print(f"[NHL][AutoRelax] min_uniques: {min_uniques} → 1")
        min_uniques2 = 1
        lineup_df = _try_build(
            players,
            cfg=cfg,
            num_lineups=num_lineups,
            min_salary=min_salary_floor,
            max_vs_goalie=new_max_vs_goalie,
            stack_requirements=stack_requirements,
            bringback=bringback,
            w_up=w_up,
            w_con=w_con,
            w_dud=w_dud,
            ownership_ctx=(has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk),
            min_uniques=min_uniques2,
            rng=rng,
        )

    if lineup_df.empty:
        # Step D: kill stack mins + bringback
        print("[NHL][AutoRelax] Disabling stacks & bringback requirements.")
        zero_stacks = {k: 0 for k in ("evline2", "evline3", "pp1_2", "pp1_3", "pp2_2", "same2", "same3", "same4", "same5")}
        lineup_df = _try_build(
            players,
            cfg=cfg,
            num_lineups=num_lineups,
            min_salary=min_salary_floor,
            max_vs_goalie=new_max_vs_goalie,
            stack_requirements=zero_stacks,
            bringback=0,
            w_up=w_up,
            w_con=w_con,
            w_dud=w_dud,
            ownership_ctx=(has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk),
            min_uniques=1,
            rng=rng,
        )

    if lineup_df.empty:
        print("[NHL][AutoRelax] Final attempt: disable enforce_min and sweep leftover targets.")

        def _final_attempt(
            players,
            cfg,
            num_lineups,
            min_salary_floor,
            max_vs_goalie,
            stack_requirements,
            bringback,
            w_up,
            w_con,
            w_dud,
            ownership_ctx,
            min_uniques,
            rng,
        ):
            has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk = ownership_ctx
            buckets_cfg = _cfg_get(cfg, ["random", "leftover_buckets"], {"mix": [0, 100, 200]})
            mode = str(_cfg_get(cfg, ["random", "leftover_mode"], "mix"))
            bucket_list = list(buckets_cfg.get(mode, [0, 100, 200]))
            ok = False
            prior_lineups: List[List[int]] = []
            lineups: List[pd.DataFrame] = []
            for k in range(num_lineups):
                ok = False
                for _lb in bucket_list:
                    ok2, idxs, extras = solve_single_lineup(
                        players,
                        min_salary=int(min_salary_floor),
                        max_vs_goalie=max_vs_goalie,
                        stack_requirements=stack_requirements,
                        bringback_min=bringback,
                        bringback_max=int(_cfg_get(cfg, ["correlation", "bringbacks_max"], 999)),
                        min_uniques=min_uniques,
                        prior_lineups=prior_lineups,
                        w_up=w_up,
                        w_con=w_con,
                        w_dud=w_dud,
                        w_eff=float(_cfg_get(cfg, ["ownership", "w_eff"], 0.0)) if ownership_enabled and has_own else 0.0,
                        w_ev3=float(_cfg_get(cfg, ["correlation", "w_ev_stack_3p"], 0.0)),
                        w_pp1=float(_cfg_get(cfg, ["correlation", "w_pp1_stack_3p"], 0.0)),
                        noise=rng.normal(0.0, float(_cfg_get(cfg, ["random", "objective_noise_std"], 0.0)), size=len(players)),
                        has_own=has_own,
                        ownership_enabled=ownership_enabled,
                        total_own_max=float(_cfg_get(cfg, ["ownership", "total_own_max"], 130.0)),
                        chalk_thresh=float(_cfg_get(cfg, ["ownership", "chalk_thresh"], 20.0)),
                        max_chalk=int(_cfg_get(cfg, ["ownership", "max_chalk"], 2)),
                    )
                    if ok2:
                        lineups.append(_finalise_lineup(players, idxs, len(prior_lineups) + 1))
                        prior_lineups.append(list(idxs))
                        ok = True
                        break
                if not ok:
                    break
            return pd.concat(lineups, ignore_index=True) if lineups else pd.DataFrame()

        lineup_df = _final_attempt(
            players,
            cfg,
            num_lineups,
            min_salary_floor=min_salary_floor,
            max_vs_goalie=new_max_vs_goalie,
            stack_requirements=zero_stacks if "zero_stacks" in locals() else stack_requirements,
            bringback=0,
            w_up=w_up,
            w_con=w_con,
            w_dud=w_dud,
            ownership_ctx=(has_own, ownership_enabled, total_own_max, chalk_thresh, max_chalk),
            min_uniques=1,
            rng=rng,
        )

    if lineup_df.empty:
        print("No lineups produced.")
        return

    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = [
        "LineupID",
        "Slot",
        "Name",
        "PosCanon",
        "Team",
        "Opp",
        "Salary",
        "Proj",
        "Own",
        "EV_LINE_TAG",
        "PP_TAG",
    ]
    for c in cols:
        if c not in lineup_df.columns:
            lineup_df[c] = None
    lineup_df[cols].to_csv(out, index=False)
    produced = lineup_df["LineupID"].nunique()
    print(f"Wrote {produced} lineups -> {out}")


if __name__ == "__main__":
    raise SystemExit("Use cli/nhl_opt.py to run the optimizer with configuration support.")
