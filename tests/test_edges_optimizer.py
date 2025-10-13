import math
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nhl_tools.nhl_optimizer import EdgesOptimizer, ShapePlan
from nhl_tools.nhl_data import _normalize_opp


def _sample_pool() -> pd.DataFrame:
    data = [
        {"Name": "A1", "Pos": "C", "Team": "AAA", "Opp": "BBB", "Salary": 6200, "Proj": 12.0, "Full": "1F", "pp_unit": 1, "IsSkater": True, "IsGoalie": False, "Ownership": 20.0},
        {"Name": "A2", "Pos": "W", "Team": "AAA", "Opp": "BBB", "Salary": 6000, "Proj": 11.5, "Full": "1F", "pp_unit": 1, "IsSkater": True, "IsGoalie": False, "Ownership": 18.0},
        {"Name": "A3", "Pos": "W", "Team": "AAA", "Opp": "BBB", "Salary": 5800, "Proj": 11.2, "Full": "1F", "pp_unit": 1, "IsSkater": True, "IsGoalie": False, "Ownership": 17.0},
        {"Name": "A4", "Pos": "D", "Team": "AAA", "Opp": "BBB", "Salary": 5200, "Proj": 10.8, "Full": "1D", "pp_unit": 1, "IsSkater": True, "IsGoalie": False, "Ownership": 15.0},
        {"Name": "B1", "Pos": "C", "Team": "BBB", "Opp": "EEE", "Salary": 5600, "Proj": 10.5, "Full": "2F", "pp_unit": 1, "IsSkater": True, "IsGoalie": False, "Ownership": 14.0},
        {"Name": "B2", "Pos": "W", "Team": "BBB", "Opp": "EEE", "Salary": 5400, "Proj": 10.2, "Full": "2F", "pp_unit": 1, "IsSkater": True, "IsGoalie": False, "Ownership": 13.0},
        {"Name": "B3", "Pos": "D", "Team": "BBB", "Opp": "EEE", "Salary": 5200, "Proj": 9.8, "Full": "2D", "pp_unit": 2, "IsSkater": True, "IsGoalie": False, "Ownership": 12.0},
        {"Name": "C1", "Pos": "W", "Team": "CCC", "Opp": "DDD", "Salary": 4800, "Proj": 9.5, "Full": "3F", "pp_unit": 2, "IsSkater": True, "IsGoalie": False, "Ownership": 8.0},
        {"Name": "G_A", "Pos": "G", "Team": "AAA", "Opp": "BBB", "Salary": 5400, "Proj": 8.5, "Full": None, "pp_unit": None, "IsSkater": False, "IsGoalie": True, "Ownership": 10.0},
        {"Name": "G_B", "Pos": "G", "Team": "BBB", "Opp": "AAA", "Salary": 5200, "Proj": 8.0, "Full": None, "pp_unit": None, "IsSkater": False, "IsGoalie": True, "Ownership": 9.0},
    ]
    df = pd.DataFrame(data)
    df["full_line_group"] = df.apply(
        lambda r: f"{r['Team']}:{r['Full']}" if pd.notna(r["Full"]) else None,
        axis=1,
    )
    df["GameKey"] = df.apply(
        lambda r: "AAA@BBB" if {r["Team"], r["Opp"]} == {"AAA", "BBB"} else "CCC@DDD",
        axis=1,
    )
    return df


def _config() -> dict:
    return {
        "stacks": {
            "preferred_team_shapes": [[4, 3, 1]],
            "primary_evline": {"min_size": 3, "max_size": 4, "required": True},
            "secondary_pp1": {"min_size": 2, "max_size": 3},
            "pp1_full_unit": {"enabled": False, "min_size": 4, "max_size": 5},
        },
        "ownership": {"enable": True, "max_sum": 300.0, "max_hhi": 0.5},
        "diversity": {"tie_break_random_pct": 0.0},
    }


def test_edges_optimizer_constraints():
    pool = _sample_pool()
    cfg = _config()
    optimizer = EdgesOptimizer(
        pool,
        cfg,
        [ShapePlan([4, 3, 1], primary_mode="ev")],
        tie_break_random_pct=0.0,
        bringbacks_enabled=False,
        max_from_opponent=0,
        forbid_goalie_conflict=True,
        salary_min=49500,
        salary_max=50000,
        ownership_limits={"max_sum": 300.0, "max_hhi": 0.5},
    )

    lineups = optimizer.build_lineups(1, projection_guard=None, require_unique=True, max_duplication=None)
    assert lineups, "Expected at least one lineup"
    lu = lineups[0]

    goalie_team = pool.loc[lu.slots["G"], "Team"]
    for idx in lu.indices:
        if pool.loc[idx, "IsSkater"]:
            assert pool.loc[idx, "Opp"] != goalie_team

    ev_size, pp1_size = optimizer._lineup_stack_sizes(lu.indices)
    assert 3 <= ev_size <= 4
    assert pp1_size >= 2
    assert lu.salary >= 49500
    if not math.isnan(lu.ownership_sum):
        assert lu.ownership_sum <= 300.0


def test_goalie_conflict_with_verbose_opp_strings():
    pool = _sample_pool().copy()
    pool.loc[pool["Team"] == "AAA", "Opp"] = pool.loc[pool["Team"] == "AAA", "Opp"].apply(
        lambda opp: f"vs {opp} - Confirmed Starter"
    )
    pool.loc[pool["Name"] == "G_A", "Opp"] = "VS BBB - Confirmed"
    pool.loc[pool["Name"] == "G_B", "Opp"] = "@ AAA (Probable Goalie)"

    pool["Opp"] = pool["Opp"].map(_normalize_opp)
    assert set(pool.loc[pool["Team"] == "AAA", "Opp"]) == {"BBB"}
    assert pool.loc[pool["Name"] == "G_B", "Opp"].iloc[0] == "AAA"

    cfg = _config()
    optimizer = EdgesOptimizer(
        pool,
        cfg,
        [ShapePlan([4, 3, 1], primary_mode="ev")],
        tie_break_random_pct=0.0,
        bringbacks_enabled=False,
        max_from_opponent=0,
        forbid_goalie_conflict=True,
        salary_min=49500,
        salary_max=50000,
        ownership_limits={"max_sum": 300.0, "max_hhi": 0.5},
    )

    lineups = optimizer.build_lineups(1, projection_guard=None, require_unique=True, max_duplication=None)
    assert lineups, "Expected at least one lineup"
    lu = lineups[0]

    goalie_team = pool.loc[lu.slots["G"], "Team"]
    for idx in lu.indices:
        if pool.loc[idx, "IsSkater"]:
            assert pool.loc[idx, "Opp"] != goalie_team
