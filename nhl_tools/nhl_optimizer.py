from __future__ import annotations
import argparse, os, sys, math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from pulp import LpProblem, LpVariable, LpBinary, LpMaximize, lpSum, LpStatusOptimal, PULP_CBC_CMD

from .nhl_data import load_labs_for_date, DK_ROSTER, DK_SALARY_CAP


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LABS_DIR = os.path.join(REPO_ROOT, "dk_data")
from .nhl_stacks import line_bucket, group_line_members, game_pairs

# ---------- objective helpers (consistency / upside / duds) ----------
def add_quality_columns(df: pd.DataFrame,
                        ceil_mult: float = 1.6,
                        floor_mult: float = 0.55) -> pd.DataFrame:
    """
    If Labs doesn't provide ceiling/floor, synthesize simple ones.
    """
    out = df.copy()
    out["Ceil"] = out["Proj"] * float(ceil_mult)
    out["Floor"]= out["Proj"] * float(floor_mult)
    # Z-scores per position for upside/consistency
    def zcol(col, by="PosCanon"):
        g = out.groupby(by)[col]
        return (out[col] - g.transform("mean")) / (g.transform("std").replace(0, np.nan))
    out["UpsideZ"] = zcol("Ceil")
    out["ConsistencyZ"] = -zcol("Floor")  # lower floor -> worse consistency; invert sign
    # Dud penalty ~ chance under 3 DK points proxy
    out["DudPenalty"] = (3.0 / out["Proj"].clip(lower=1e-3)).clip(upper=2.0)
    return out

# ---------- linear optimizer per lineup ----------
def _flatten_history_entry(entry):
    """Yield scalar items from possibly nested history containers."""
    if entry is None:
        return
    if isinstance(entry, (pd.Series, np.ndarray)):
        entry = entry.tolist()
    if isinstance(entry, (set, list, tuple)):
        for sub in entry:
            yield from _flatten_history_entry(sub)
    else:
        yield entry


def _normalise_history(df: pd.DataFrame, idx: range, diversify_with):
    """Convert historical lineup representations into skater index sets."""
    if not diversify_with:
        return

    if isinstance(diversify_with, (list, tuple, set)):
        raw_history = list(diversify_with)
    else:
        raw_history = [diversify_with]

    name_to_idx: dict[str, set[int]] = {}
    for i in idx:
        if df.loc[i, "PosCanon"] == "G":
            continue
        name_to_idx.setdefault(str(df.loc[i, "Name"]), set()).add(i)

    for prev in raw_history:
        forbid_idx: set[int] = set()
        for item in _flatten_history_entry(prev):
            if isinstance(item, (int, np.integer)):
                j = int(item)
                if 0 <= j < len(df) and df.loc[j, "PosCanon"] != "G":
                    forbid_idx.add(j)
            elif isinstance(item, str):
                forbid_idx.update(name_to_idx.get(item, set()))
            elif isinstance(item, tuple) and item:
                name = str(item[0])
                forbid_idx.update(name_to_idx.get(name, set()))
        if forbid_idx:
            yield forbid_idx


def solve_single_lineup(players: pd.DataFrame,
                        w_up: float, w_con: float, w_dud: float,
                        min_salary: int, max_vs_goalie: int,
                        need_ev2: int, need_ev3: int,
                        need_pp1_2: int, need_pp1_3: int, need_pp2_2: int,
                        need_same2: int, need_same3: int, need_same4: int, need_same5: int,
                        need_bringback: int,
                        diversify: int,
                        diversify_with: list[list[str]] | list[str] | None = None) -> Tuple[bool, List[int]]:
    """
    Build one DK lineup. Returns (ok, idx_list)
    """
    df = players.reset_index(drop=True).copy()
    N = len(df)
    # index sets
    idx = range(N)
    C = [i for i in idx if df.loc[i,"PosCanon"]=="C"]
    W = [i for i in idx if df.loc[i,"PosCanon"]=="W"]
    D = [i for i in idx if df.loc[i,"PosCanon"]=="D"]
    G = [i for i in idx if df.loc[i,"PosCanon"]=="G"]
    SK = [i for i in idx if i not in G]

    # variables
    x = LpVariable.dicts("x", idx, lowBound=0, upBound=1, cat=LpBinary)
    # UTIL helper: we'll enforce total count == 9 with each position minima
    prob = LpProblem("DK_NHL", LpMaximize)

    # projection with quality modifiers
    score = df["Proj"] * (1.0 + w_up*df["UpsideZ"].fillna(0.0)) * (1.0 + w_con*df["ConsistencyZ"].fillna(0.0)) \
            - w_dud*df["DudPenalty"].fillna(0.0)
    prob += lpSum(x[i] * float(score.loc[i]) for i in idx)

    # salary
    prob += lpSum(x[i] * int(df.loc[i,"Salary"]) for i in idx) <= DK_SALARY_CAP
    prob += lpSum(x[i] * int(df.loc[i,"Salary"]) for i in idx) >= int(min_salary)

    # roster counts
    # DraftKings uses an eight-player positional core (2C/3W/2D/1G) plus a UTIL
    # skater.  The FantasyLabs exports only list a single canonical position per
    # player, so the UTIL spot must be satisfied by exceeding one of the skater
    # minimums rather than relying on multi-position eligibility.  Using equality
    # for the skater counts makes the model infeasible because 2+3+2+1=8 while we
    # also demand nine total players.  Relax the skater constraints to minimums
    # while keeping the goalie exact and the overall roster size at nine.
    prob += lpSum(x[i] for i in C) >= DK_ROSTER["C"]
    prob += lpSum(x[i] for i in W) >= DK_ROSTER["W"]
    prob += lpSum(x[i] for i in D) >= DK_ROSTER["D"]
    prob += lpSum(x[i] for i in G) == DK_ROSTER["G"]
    prob += lpSum(x[i] for i in idx) == sum(DK_ROSTER.values())  # total 9

    # no more than 5 skaters from same team by default (tweakable by editing)
    for team, cnt in df.loc[SK,"Team"].value_counts().items():
        Ti = [i for i in SK if df.loc[i,"Team"] == team]
        prob += lpSum(x[i] for i in Ti) <= 5

    # goalie anti-corr
    for gi in G:
        gteam = df.loc[gi,"Team"]
        opp   = df.loc[gi,"Opp"]
        if not isinstance(opp,str): continue
        vs = [i for i in SK if df.loc[i,"Team"] == opp]
        prob += lpSum(x[i] for i in vs) <= int(max_vs_goalie) + (1 - x[gi])*9  # relax if goalie not chosen

    # EV line and PP groupings
    df = line_bucket(df)
    groups = {}
    for i,r in df.iterrows():
        if r.get("EV_LINE_TAG"): groups.setdefault(("EV", r["EV_LINE_TAG"]), []).append(i)
        if r.get("PP_TAG"):      groups.setdefault(("PP", r["PP_TAG"]), []).append(i)

    # activation binaries to count stacks present
    y_ev2 = [LpVariable(f"y_ev2_{k[1]}", 0,1, LpBinary) for k,g in groups.items() if k[0]=="EV"]
    ev2_keys = [k for k in groups.keys() if k[0]=="EV"]
    y_pp1_2 = [LpVariable(f"y_pp1_2_{k[1]}",0,1,LpBinary) for k in groups.keys() if k[0]=="PP" and k[1].endswith("PP1")]
    pp1_2_keys = [k for k in groups.keys() if k[0]=="PP" and k[1].endswith("PP1")]
    y_pp2_2 = [LpVariable(f"y_pp2_2_{k[1]}",0,1,LpBinary) for k in groups.keys() if k[0]=="PP" and k[1].endswith("PP2")]
    pp2_2_keys = [k for k in groups.keys() if k[0]=="PP" and k[1].endswith("PP2")]

    # EV 2+ and 3+
    for y, k in zip(y_ev2, ev2_keys):
        members = groups[k]
        prob += lpSum(x[i] for i in members) >= 2*y
    prob += lpSum(y_ev2) >= int(need_ev2)

    # EV 3+ needs own activations (ensure >=3)
    y_ev3 = [LpVariable(f"y_ev3_{k[1]}",0,1,LpBinary) for k in ev2_keys]
    for y, k in zip(y_ev3, ev2_keys):
        members = groups[k]
        prob += lpSum(x[i] for i in members) >= 3*y
    prob += lpSum(y_ev3) >= int(need_ev3)

    # PP1 2+ / 3+
    y_pp1_3 = [LpVariable(f"y_pp1_3_{k[1]}",0,1,LpBinary) for k in pp1_2_keys]
    for y2,y3,k in zip(y_pp1_2, y_pp1_3, pp1_2_keys):
        mem = groups[k]
        prob += lpSum(x[i] for i in mem) >= 2*y2
        prob += lpSum(x[i] for i in mem) >= 3*y3
    prob += lpSum(y_pp1_2) >= int(need_pp1_2)
    prob += lpSum(y_pp1_3) >= int(need_pp1_3)

    # PP2 2+
    for y,k in zip(y_pp2_2, pp2_2_keys):
        mem = groups[k]
        prob += lpSum(x[i] for i in mem) >= 2*y
    prob += lpSum(y_pp2_2) >= int(need_pp2_2)

    # SameTeam N+ counts (skaters only)
    # Count per team via big-M trick
    teams = sorted(df["Team"].dropna().unique().tolist())
    y_same2 = [LpVariable(f"y_same2_{t}",0,1,LpBinary) for t in teams]
    y_same3 = [LpVariable(f"y_same3_{t}",0,1,LpBinary) for t in teams]
    y_same4 = [LpVariable(f"y_same4_{t}",0,1,LpBinary) for t in teams]
    y_same5 = [LpVariable(f"y_same5_{t}",0,1,LpBinary) for t in teams]
    for t, y2,y3,y4,y5 in zip(teams, y_same2,y_same3,y_same4,y_same5):
        Ti = [i for i in range(N) if (df.loc[i,"Team"]==t and df.loc[i,"PosCanon"]!="G")]
        prob += lpSum(x[i] for i in Ti) >= 2*y2
        prob += lpSum(x[i] for i in Ti) >= 3*y3
        prob += lpSum(x[i] for i in Ti) >= 4*y4
        prob += lpSum(x[i] for i in Ti) >= 5*y5
    prob += lpSum(y_same2) >= int(need_same2)
    prob += lpSum(y_same3) >= int(need_same3)
    prob += lpSum(y_same4) >= int(need_same4)
    prob += lpSum(y_same5) >= int(need_same5)

    # BringBack_1+ (game-level approximation)
    # If we have a team with >=2 skaters in game g, require >=1 skater from its opponent (for y_bb_g=1).
    games = game_pairs(df)
    y_bb = [LpVariable(f"y_bb_{a}_{b}", 0, 1, LpBinary) for a, b in games]
    y_stack_a = [LpVariable(f"y_stack_{a}_{b}_{a}", 0, 1, LpBinary) for a, b in games]
    y_stack_b = [LpVariable(f"y_stack_{a}_{b}_{b}", 0, 1, LpBinary) for a, b in games]
    for (a, b), y_game, ya, yb in zip(games, y_bb, y_stack_a, y_stack_b):
        A = [i for i in SK if str(df.loc[i, "Team"]).upper() == a]
        B = [i for i in SK if str(df.loc[i, "Team"]).upper() == b]
        if not A or not B:
            # Without skaters from both teams, the bring-back constraint is moot.
            prob += y_game == 0
            prob += ya == 0
            prob += yb == 0
            continue

        prob += lpSum(x[i] for i in A) >= 2 * ya
        prob += lpSum(x[i] for i in B) >= ya

        prob += lpSum(x[i] for i in B) >= 2 * yb
        prob += lpSum(x[i] for i in A) >= yb

        # tie the game activation to either side triggering the condition
        prob += y_game <= ya + yb
        prob += y_game >= ya
        prob += y_game >= yb
    prob += lpSum(y_bb) >= int(need_bringback)

    # diversification (soft): forbid all players from previous lineup
    if diversify > 0 and diversify_with:
        for forbid_idx in _normalise_history(df, idx, diversify_with):
            limit = max(0, len(forbid_idx) - int(diversify))
            prob += lpSum(x[i] for i in forbid_idx) <= limit

    # solve
    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        return (False, [])
    chosen = [i for i in idx if x[i].value() >= 0.99]
    return (True, chosen)

def build_lineups(df: pd.DataFrame, n: int,
                  w_up: float, w_con: float, w_dud: float,
                  min_salary: int, max_vs_goalie: int,
                  need_ev2: int, need_ev3: int,
                  need_pp1_2: int, need_pp1_3: int, need_pp2_2: int,
                  need_same2: int, need_same3: int, need_same4: int, need_same5: int,
                  need_bringback: int,
                  diversify: int) -> pd.DataFrame:
    """
    Greedy generate N lineups; after each solve, nudge diversification by forbidding the exact set.
    """
    players = add_quality_columns(df)
    outs = []
    history: list[list[int]] = []
    for k in range(n):
        ok, idxs = solve_single_lineup(
            players, w_up, w_con, w_dud, min_salary, max_vs_goalie,
            need_ev2, need_ev3, need_pp1_2, need_pp1_3, need_pp2_2,
            need_same2, need_same3, need_same4, need_same5,
            need_bringback,
            diversify,
            diversify_with=(history if diversify>0 else None)
        )
        if not ok: break
        chosen = players.loc[idxs].copy()
        # record
        def slot_order_key(r):
            order = dict(C=1,W=2,D=3,G=4)
            return (order.get(r["PosCanon"],9), -float(r["Proj"]), r["Name"])
        chosen = chosen.sort_values(by=["PosCanon","Proj","Name"], key=lambda c: c.map(lambda _: 0))
        # Fill DK slots in a simple greedy way
        C = chosen[chosen["PosCanon"]=="C"].head(2)
        W = chosen[chosen["PosCanon"]=="W"].head(3)
        D = chosen[chosen["PosCanon"]=="D"].head(2)
        G = chosen[chosen["PosCanon"]=="G"].head(1)
        UTIL = chosen.drop(C.index.union(W.index).union(D.index).union(G.index)).head(1)

        lineup = pd.concat([C,W,D,G,UTIL])
        lineup["LineupID"] = k+1
        lineup["Slot"] = (["C1","C2","W1","W2","W3","D1","D2","G","UTIL"])[:len(lineup)]
        outs.append(lineup)
        history.append([int(i) for i in idxs if players.loc[i, "PosCanon"] != "G"])
    if outs:
        return pd.concat(outs, ignore_index=True)
    return pd.DataFrame()

def main():
    ap = argparse.ArgumentParser(description="NHL Optimizer (DK) — FantasyLabs input (separate from NFL).")
    ap.add_argument("--labs-dir", default=DEFAULT_LABS_DIR,
                    help="Folder with FantasyLabs NHL CSVs (default: %(default)s)")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", required=True, help="Output CSV for lineups")
    ap.add_argument("--num-lineups", type=int, default=20)
    ap.add_argument("--min-salary", type=int, default=49500)
    ap.add_argument("--max-vs-goalie", type=int, default=0)

    # stacks
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

    # quality weights
    ap.add_argument("--w-up", type=float, default=0.15)
    ap.add_argument("--w-con", type=float, default=0.05)
    ap.add_argument("--w-dud", type=float, default=0.03)
    ap.add_argument("--ceil-mult", type=float, default=1.6)
    ap.add_argument("--floor-mult", type=float, default=0.55)

    ap.add_argument("--diversify", type=int, default=1, help=">0 to force at least one change from prior lineup")
    args = ap.parse_args()

    df = load_labs_for_date(args.labs_dir, args.date)
    lineups = build_lineups(
        df, args.num_lineups,
        args.w_up, args.w_con, args.w_dud,
        args.min_salary, args.max_vs_goalie,
        args.evline2, args.evline3,
        args.pp1_2, args.pp1_3, args.pp2_2,
        args.same2, args.same3, args.same4, args.same5,
        args.bringback,
        args.diversify
    )
    if lineups.empty:
        print("No lineups produced.")
        sys.exit(2)

    # Write DK-like CSV (one row per player per lineup) + summary per lineup at bottom
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["LineupID","Slot","Name","PosCanon","Team","Opp","Salary","Proj","Own","EV_LINE_TAG","PP_TAG"]
    for c in cols:
        if c not in lineups.columns: lineups[c]=None
    lineups[cols].to_csv(args.out, index=False)
    print(f"Wrote {len(lineups['LineupID'].unique())} lineups -> {args.out}")

if __name__ == "__main__":
    main()
