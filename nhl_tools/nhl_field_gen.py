from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import pandas as pd

# -----------------------------
# DK NHL roster + cap settings
# -----------------------------
DK_SALARY_CAP = 50000
DK_ROSTER = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
POS_SLOTS = {
    "C1": ["C"],
    "C2": ["C"],
    "W1": ["W"],
    "W2": ["W"],
    "W3": ["W"],
    "D1": ["D"],
    "D2": ["D"],
    "G":  ["G"],
    "UTIL": ["C", "W", "D"],  # UTIL cannot be G on DK
}

# -----------------------------
# Stack templates (primary-secondary-filler)
# Interpret as counts of SKATERS (no goalie)
# -----------------------------
DEFAULT_STACK_TEMPLATES: List[Tuple[int, int, int]] = [
    (3, 2, 2),  # classic 3-2-2
    (3, 3, 1),  # 3-3-1
    (4, 2, 1),  # 4-2-1
    (2, 2, 3),  # 2-2-3 (more balanced)
    (3, 2, 1),  # 3-2 + singles
]

DEFAULT_STACK_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


@dataclass
class FieldGenConfig:
    field_size: int = 5000
    max_skaters_same_team: int = 5                # DK allows up to 5 skaters from one team
    lock_no_goalie_vs_opp: bool = True            # forbid goalie vs any opposing skater
    min_salary: int = 48000
    max_salary: int = 50000
    # Sampling: probabilities ~ (ownership**beta) * exp(temp * projection)
    sample_beta_own: float = 1.0
    sample_temp_proj: float = 0.03
    # Randomness/jitter in projections to diversify:
    proj_jitter_sd: float = 2.5
    # Stacks
    stack_templates: List[Tuple[int, int, int]] = None
    stack_weights: List[float] = None
    # Uniqueness
    max_attempts_per_lineup: int = 400
    random_seed: Optional[int] = 777


# -----------------------------
# Utilities to normalize inputs
# -----------------------------
def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}
    def get(*names, default=None):
        for nm in names:
            if nm.lower() in cols:
                return cols[nm.lower()]
        return default

    name_col = get("Name", "player", "Player Name", "player_name", default="Name")
    id_col = get("ID", "id", "player_id", default="ID")
    team_col = get("Team", "team", "tm", default="Team")
    opp_col  = get("Opp", "opponent", "opp", default="Opp")
    pos_col  = get("Roster Position", "roster_position", "pos", "position", default="Roster Position")
    sal_col  = get("Salary", "salary", default="Salary")
    proj_col = get("Projection", "Proj", "My Proj", "proj", "points", default="Projection")
    own_col  = get("Ownership", "Own%", "My Own", "own", "ownership", default=None)

    df.rename(columns={
        name_col: "Name",
        id_col: "ID",
        team_col: "Team",
        opp_col: "Opp",
        pos_col: "Pos",
        sal_col: "Salary"
    }, inplace=True)

    if proj_col and proj_col in df.columns:
        df.rename(columns={proj_col: "Projection"}, inplace=True)
    else:
        df["Projection"] = 0.0

    if own_col and own_col in df.columns:
        df.rename(columns={own_col: "Ownership"}, inplace=True)
        df["Ownership"] = pd.to_numeric(df["Ownership"], errors="coerce").fillna(0.0)
        if df["Ownership"].max() > 1.0:
            df["Ownership"] = df["Ownership"] / 100.0
    else:
        df["Ownership"] = 0.0

    def split_pos(p):
        if pd.isna(p):
            return []
        s = str(p).upper().replace("+", "/").replace(",", "/")
        return [t.strip() for t in s.split("/") if t.strip()]

    df["PosList"] = df["Pos"].map(split_pos)
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
    df["Projection"] = pd.to_numeric(df["Projection"], errors="coerce").fillna(0.0)
    df["Ownership"] = pd.to_numeric(df["Ownership"], errors="coerce").fillna(0.0)

    return df[["Name", "ID", "Team", "Opp", "Pos", "PosList", "Salary", "Projection", "Ownership"]]


def _prob_score(row, beta: float, temp: float) -> float:
    own = max(float(row["Ownership"]), 0.0)
    proj = float(row["Projection"])
    return (own ** beta) * math.exp(temp * proj)


def _sample(players: pd.DataFrame, k: int, pos_allow: Iterable[str],
            team: Optional[str], forbid_opp: Optional[str],
            salary_left: int, max_per_team: Dict[str, int],
            taken_ids: set,
            cfg: FieldGenConfig) -> List[int]:
    eligible = players.index.tolist()
    choice = []
    tries = 0

    while len(choice) < k and tries < cfg.max_attempts_per_lineup:
        tries += 1
        mask = players["ID"].astype(str).apply(lambda pid: pid not in taken_ids)
        mask &= players["PosList"].apply(lambda L: any(p in L for p in pos_allow))
        if team:
            mask &= (players["Team"] == team)
        if forbid_opp:
            mask &= (players["Team"] != forbid_opp)

        pool = players[mask]
        if pool.empty:
            break

        probs = pool.apply(lambda r: _prob_score(r, cfg.sample_beta_own, cfg.sample_temp_proj), axis=1)
        jitter = pd.Series([random.gauss(0, cfg.proj_jitter_sd) for _ in range(len(pool))], index=pool.index)
        probs = probs * (1.0 + 0.01 * jitter.clip(-3, 3))

        s = probs.sum()
        if s <= 0:
            probs = pd.Series([1.0] * len(pool), index=pool.index)
            s = probs.sum()
        probs = probs / s

        pick = random.choices(pool.index.tolist(), weights=probs.values.tolist(), k=1)[0]
        row = players.loc[pick]

        team_now = row["Team"]
        if max_per_team.get(team_now, 0) <= 0:
            continue
        if row["Salary"] > salary_left:
            continue

        choice.append(pick)
        taken_ids.add(str(row["ID"]))
        max_per_team[team_now] = max_per_team.get(team_now, 0) - 1

    return choice


def _choose_two_teams(players: pd.DataFrame) -> Tuple[str, str]:
    agg = players.groupby("Team").apply(
        lambda g: (g["Projection"] * (1.0 + 2.0 * g["Ownership"])).sum()
    )
    if len(agg) < 2:
        teams = list(players["Team"].dropna().unique())
        if len(teams) == 1:
            return teams[0], teams[0]
        return teams[0], teams[1]
    probs = (agg - agg.min() + 1e-6)
    probs = probs / probs.sum()
    prim = random.choices(agg.index.tolist(), weights=probs.values.tolist(), k=1)[0]
    sec_pool = agg.drop(index=prim)
    sec_probs = (sec_pool - sec_pool.min() + 1e-6)
    sec_probs = sec_probs / sec_probs.sum()
    sec = random.choices(sec_pool.index.tolist(), weights=sec_probs.values.tolist(), k=1)[0]
    return prim, sec


def _fill_position(players: pd.DataFrame, slot: str, taken_ids: set,
                   salary_left: int, per_team_left: Dict[str, int],
                   forbid_opp_for_skaters: Optional[str],
                   cfg: FieldGenConfig) -> Optional[int]:
    if slot == "G":
        mask = players["PosList"].apply(lambda L: "G" in L)
        pool = players[mask].copy()
        if pool.empty:
            return None
        if forbid_opp_for_skaters:
            pool = pool[pool["Team"] != forbid_opp_for_skaters]
        pool = pool[pool["ID"].astype(str).apply(lambda x: x not in taken_ids)]
        pool = pool[pool["Salary"] <= salary_left]
        pool = pool[pool["Team"].apply(lambda t: per_team_left.get(t, 0) > 0)]
        if pool.empty:
            return None
        probs = pool.apply(lambda r: _prob_score(r, cfg.sample_beta_own, cfg.sample_temp_proj), axis=1)
        s = probs.sum()
        if s <= 0:
            probs[:] = 1.0
            s = probs.sum()
        probs = probs / s
        pick = random.choices(pool.index.tolist(), weights=probs.values.tolist(), k=1)[0]
        return pick

    pos_allow = POS_SLOTS[slot]
    forbid_opp = forbid_opp_for_skaters if cfg.lock_no_goalie_vs_opp else None
    choice = _sample(players, 1, pos_allow, team=None, forbid_opp=forbid_opp,
                     salary_left=salary_left, max_per_team=per_team_left,
                     taken_ids=taken_ids, cfg=cfg)
    return choice[0] if choice else None


def _lineup_salary(players: pd.DataFrame, idxs: List[int]) -> int:
    return int(players.loc[idxs]["Salary"].sum())


def _lineup_valid(players: pd.DataFrame, idxs: List[int], cfg: FieldGenConfig) -> bool:
    if len(idxs) != len(DK_ROSTER):
        return False
    sal = _lineup_salary(players, idxs)
    if not (cfg.min_salary <= sal <= cfg.max_salary):
        return False
    if cfg.lock_no_goalie_vs_opp:
        g_idx = idxs[DK_ROSTER.index("G")]
        g_tm = players.loc[g_idx, "Team"]
        opp_teams = set(players.loc[i, "Team"] for n,i in zip(DK_ROSTER, idxs) if n != "G")
        if g_tm in opp_teams:
            return False
    return True


def _choose_skaters_for_team(players: pd.DataFrame, team: str, k: int,
                             taken_ids: set, per_team_left: Dict[str, int],
                             forbid_opp: Optional[str], salary_left: int,
                             cfg: FieldGenConfig) -> List[int]:
    mask = (players["Team"] == team) & (players["PosList"].apply(lambda L: "G" not in L))
    return _sample(players[mask], k, pos_allow=["C", "W", "D"],
                   team=None, forbid_opp=forbid_opp, salary_left=salary_left,
                   max_per_team=per_team_left, taken_ids=taken_ids, cfg=cfg)


def _final_fill(players: pd.DataFrame, idxs: Dict[str, int], cfg: FieldGenConfig) -> Optional[List[int]]:
    taken_ids = set(str(players.loc[i, "ID"]) for i in idxs.values())
    per_team_left: Dict[str, int] = {}
    for t, count in players.groupby("Team").size().items():
        per_team_left[t] = cfg.max_skaters_same_team - sum(
            1 for _n, i in idxs.items() if players.loc[i, "Team"] == t and "G" not in players.loc[i, "PosList"]
        )

    current_idxs = list(idx for idx in idxs.values())
    salary_left = DK_SALARY_CAP - _lineup_salary(players, current_idxs)

    forbid_opp_for_skaters = None
    if "G" in idxs:
        g_idx = idxs["G"]
        opp = players.loc[g_idx, "Opp"] if "Opp" in players.columns else None
        forbid_opp_for_skaters = opp

    filled = dict(idxs)
    for slot in DK_ROSTER:
        if slot in filled:
            continue
        pick = _fill_position(players, slot, taken_ids, salary_left, per_team_left,
                              forbid_opp_for_skaters, cfg)
        if pick is None:
            return None
        filled[slot] = pick
        taken_ids.add(str(players.loc[pick, "ID"]))
        t = players.loc[pick, "Team"]
        if "G" not in players.loc[pick, "PosList"]:
            per_team_left[t] = per_team_left.get(t, cfg.max_skaters_same_team) - 1
        salary_left = DK_SALARY_CAP - _lineup_salary(players, list(filled.values()))

    idx_ordered = [filled[s] for s in DK_ROSTER]
    if not _lineup_valid(players, idx_ordered, cfg):
        return None
    return idx_ordered


def generate_field_lineups(players_df: pd.DataFrame, cfg: FieldGenConfig) -> pd.DataFrame:
    if cfg.stack_templates is None:
        cfg.stack_templates = DEFAULT_STACK_TEMPLATES
    if cfg.stack_weights is None:
        cfg.stack_weights = DEFAULT_STACK_WEIGHTS

    random.seed(cfg.random_seed)

    players = _norm_cols(players_df)
    lineups: List[Dict[str, str]] = []
    seen_sets: set = set()

    attempts = 0
    target = int(cfg.field_size)

    while len(lineups) < target and attempts < target * cfg.max_attempts_per_lineup:
        attempts += 1

        stack_shape = random.choices(cfg.stack_templates, weights=cfg.stack_weights, k=1)[0]
        k_prim, k_sec, k_fill = stack_shape

        prim, sec = _choose_two_teams(players)

        per_team_left = {t: cfg.max_skaters_same_team for t in players["Team"].unique()}

        idxs: Dict[str, int] = {}

        g_mask = players["PosList"].apply(lambda L: "G" in L)
        g_pool = players[g_mask].copy()
        if g_pool.empty:
            break

        g_probs = g_pool.apply(lambda r: _prob_score(r, cfg.sample_beta_own, cfg.sample_temp_proj), axis=1)
        s = g_probs.sum()
        if s <= 0:
            g_probs[:] = 1.0
            s = g_probs.sum()
        g_probs = g_probs / s
        g_idx = random.choices(g_pool.index.tolist(), weights=g_probs.values.tolist(), k=1)[0]
        idxs["G"] = g_idx

        g_salary = players.loc[g_idx, "Salary"]
        taken_ids = {str(players.loc[g_idx, "ID"])}

        prim_pick = _choose_skaters_for_team(
            players, prim, k_prim, taken_ids, per_team_left,
            forbid_opp=None, salary_left=DK_SALARY_CAP - g_salary, cfg=cfg
        )
        sec_pick = _choose_skaters_for_team(
            players, sec, k_sec, taken_ids, per_team_left,
            forbid_opp=None,
            salary_left=DK_SALARY_CAP - (g_salary + players.loc[prim_pick]["Salary"].sum() if prim_pick else 0),
            cfg=cfg
        )

        if len(prim_pick) != k_prim or len(sec_pick) != k_sec:
            continue

        assigned: Dict[str, int] = {"G": g_idx}

        def assign_group(pick_idxs: List[int], slot_order: List[str]) -> bool:
            left = pick_idxs[:]
            for slot in slot_order:
                if slot in assigned:
                    continue
                if not left:
                    break
                found_idx = None
                for i in range(len(left)):
                    idx = left[i]
                    if any(p in players.loc[idx, "PosList"] for p in POS_SLOTS[slot]):
                        found_idx = i
                        break
                if found_idx is not None:
                    assigned[slot] = left.pop(found_idx)
            return len(left) == 0

        prim_ok = assign_group(prim_pick, ["W1","W2","W3","C1","C2","D1","D2","UTIL"])
        sec_ok  = assign_group(sec_pick,  ["W1","W2","W3","C1","C2","D1","D2","UTIL"])
        if not prim_ok or not sec_ok:
            continue

        filled_idxs = _final_fill(players, assigned, cfg)
        if filled_idxs is None:
            continue

        id_set = tuple(sorted(str(players.loc[i, "ID"]) for i in filled_idxs))
        if id_set in seen_sets:
            continue
        seen_sets.add(id_set)

        row: Dict[str, str] = {}
        salary_total = players.loc[filled_idxs]["Salary"].sum()
        for slot, idx in zip(DK_ROSTER, filled_idxs):
            row[slot] = f'{players.loc[idx, "Name"]} ({players.loc[idx, "ID"]})'
        row["Salary"] = int(salary_total)
        row["stack_shape"] = f"{k_prim}-{k_sec}-{k_fill}"
        row["primary_team"] = prim
        row["secondary_team"] = sec
        lineups.append(row)

    if len(lineups) < cfg.field_size:
        print(f"[field_gen] Warning: only generated {len(lineups)} / {cfg.field_size} lineups after {attempts} attempts.")

    return pd.DataFrame(lineups, columns=DK_ROSTER + ["Salary","stack_shape","primary_team","secondary_team"])
