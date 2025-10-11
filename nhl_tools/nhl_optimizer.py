#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

def add_quality_columns(df: pd.DataFrame, ceil_mult: float = 1.6, floor_mult: float = 0.55) -> pd.DataFrame:
    out = df.copy()
    out["Ceil"] = out["Proj"] * float(ceil_mult)
    out["Floor"] = out["Proj"] * float(floor_mult)


        g = out.groupby(by)[col]
        return (out[col] - g.transform("mean")) / (g.transform("std").replace(0, np.nan))

    out["UpsideZ"] = zcol("Ceil")
    out["ConsistencyZ"] = -zcol("Floor")

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

) -> Tuple[bool, List[int]]:
    df = players.reset_index(drop=True).copy()
    N = len(df)
    idx = range(N)
    C = [i for i in idx if df.loc[i, "PosCanon"] == "C"]
    W = [i for i in idx if df.loc[i, "PosCanon"] == "W"]
    D = [i for i in idx if df.loc[i, "PosCanon"] == "D"]
    G = [i for i in idx if df.loc[i, "PosCanon"] == "G"]

    score = (
        df["Proj"]
        * (1.0 + w_up * df["UpsideZ"].fillna(0.0))
        * (1.0 + w_con * df["ConsistencyZ"].fillna(0.0))
        - w_dud * df["DudPenalty"].fillna(0.0)
    )
    prob += lpSum(x[i] * float(score.loc[i]) for i in idx)



    prob += lpSum(x[i] for i in C) >= DK_ROSTER_COUNTS["C"]
    prob += lpSum(x[i] for i in W) >= DK_ROSTER_COUNTS["W"]
    prob += lpSum(x[i] for i in D) >= DK_ROSTER_COUNTS["D"]
    prob += lpSum(x[i] for i in G) == DK_ROSTER_COUNTS["G"]

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

    status = prob.solve(PULP_CBC_CMD(msg=False))
    if status != LpStatusOptimal:
        return (False, [])
    chosen = [i for i in idx if x[i].value() and x[i].value() >= 0.99]
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

        )
        if not ok:
            break
        chosen = players.loc[idxs].copy()

        outs.append(lineup)
        history.append(lineup["Name"].tolist())
    if outs:
        return pd.concat(outs, ignore_index=True)
    return pd.DataFrame()



    return out_path


if __name__ == "__main__":
    main()

