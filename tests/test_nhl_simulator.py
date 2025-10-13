import math

import numpy as np
import pandas as pd

from nhl_tools.nhl_simulator import (
    DK_SLOTS,
    Lineup,
    PlayerRecord,
    _build_lineups,
    _format_player_cell,
    _prepare_sim,
    _run_simulation,
)


def _make_player(
    name: str,
    position: str,
    team: str,
    salary: float,
    projection: float,
    *,
    opp: str | None = None,
    full: str | None = None,
    pp_unit: int | None = None,
    ownership: float | None = None,
) -> PlayerRecord:
    return PlayerRecord(
        name=name,
        position=position,
        team=team,
        opp=opp,
        salary=salary,
        projection=projection,
        ceiling=projection,
        actual=0.0,
        ownership=ownership,
        full=full,
        pp_unit=pp_unit,
    )


def _lineup_from_players(lineup_id: str, slot_players: dict[str, PlayerRecord]) -> Lineup:
    # Ensure every DraftKings slot is present for deterministic column ordering
    normalized_slots = {slot: slot_players.get(slot) for slot in DK_SLOTS if slot_players.get(slot)}
    lineup = Lineup(
        idx=0,
        lineup_id=lineup_id,
        slot_players=normalized_slots,
        signature=f"sig-{lineup_id}",
        metrics={"weight": 1.0},
        base_score=0.0,
    )
    return lineup


def test_simulation_generates_clusters_and_wins():
    skater_a = _make_player("Skater A", "C", "AAA", 6200, 12.5, opp="BBB", full="1F", pp_unit=1)
    skater_b = _make_player("Skater B", "W", "AAA", 6100, 11.0, opp="BBB", full="1F", pp_unit=1)
    skater_c = _make_player("Skater C", "W", "AAA", 5800, 10.0, opp="BBB", full="1F", pp_unit=2)
    goalie = _make_player("Goalie D", "G", "BBB", 7900, 8.5, opp="AAA")

    lineup = _lineup_from_players(
        "1",
        {
            "C1": skater_a,
            "W1": skater_b,
            "W2": skater_c,
            "G": goalie,
        },
    )

    _prepare_sim([lineup], cfg={})
    results = _run_simulation([lineup], {lineup.signature: 1}, 100, 1000, np.random.default_rng(123))

    lineup_row = results.lineup_table.iloc[0]
    assert lineup_row["EV Cluster"] >= 3
    assert lineup_row["PP1 Cluster"] >= 2
    assert lineup_row["Win%"] > 0
    assert lineup_row["Sim Own%"] > 0


def test_simulation_handles_missing_ownership():
    skater_a = _make_player("Alpha", "C", "AAA", 5000, 9.0, full="1F")
    skater_b = _make_player("Beta", "W", "AAA", 4800, 8.2, full="1F")
    goalie = _make_player("Gamma", "G", "BBB", 7700, 7.5, opp="AAA")

    lineup = _lineup_from_players(
        "2",
        {
            "C1": skater_a,
            "W1": skater_b,
            "G": goalie,
        },
    )

    _prepare_sim([lineup], cfg={})
    results = _run_simulation([lineup], {lineup.signature: 1}, 50, 500, np.random.default_rng(321))

    lineup_row = results.lineup_table.iloc[0]
    assert np.isnan(lineup_row["Own Sum"])

    player_table = results.player_table
    assert "Proj. Own%" in player_table.columns
    assert player_table["Proj. Own%"].isna().all()


def test_build_lineups_extracts_slot_metadata_and_scores():
    df = pd.DataFrame(
        {
            "LineupID": [1],
            "C1 Name": ["Alpha"],
            "C1 Team": ["AAA"],
            "C1 Pos": ["C"],
            "C1 Salary": [5200],
            "C1 Proj": ["9.5"],
            "C1 ID": [12345],
            "C2 Name": ["Bravo"],
            "C2 Team": ["AAA"],
            "C2 Pos": ["C"],
            "C2 Salary": [5100],
            "C2 Proj": ["9.0"],
            "W1 Name": ["Charlie"],
            "W1 Team": ["AAA"],
            "W1 Pos": ["W"],
            "W1 Salary": [4900],
            "W1 Proj": ["8.6"],
            "W1 ID": ["22222"],
            "W2 Name": ["Delta"],
            "W2 Team": ["AAA"],
            "W2 Pos": ["W"],
            "W2 Salary": [4700],
            "W2 Proj": ["8.2"],
            "W3 Name": ["Echo"],
            "W3 Team": ["AAA"],
            "W3 Pos": ["W"],
            "W3 Salary": [4500],
            "W3 Proj": ["7.5"],
            "D1 Name": ["Foxtrot"],
            "D1 Team": ["AAA"],
            "D1 Pos": ["D"],
            "D1 Salary": [4400],
            "D1 Proj": ["7.2"],
            "D1 ID": ["00333"],
            "D2 Name": ["Golf"],
            "D2 Team": ["AAA"],
            "D2 Pos": ["D"],
            "D2 Salary": [4300],
            "D2 Proj": ["6.9"],
            "G Name": ["Hotel"],
            "G Team": ["BBB"],
            "G Pos": ["G"],
            "G Salary": [7800],
            "G Proj": ["8.0"],
            "UTIL Name": ["India"],
            "UTIL Team": ["AAA"],
            "UTIL Pos": ["W"],
            "UTIL Salary": [3600],
            "UTIL Proj": ["5.1"],
            "UTIL2 Name": ["Juliet"],
            "UTIL2 Team": ["AAA"],
            "UTIL2 Pos": ["C"],
            "UTIL2 Salary": [3500],
            "UTIL2 Proj": ["4.9"],
            "UTIL2 ID": [np.nan],
        }
    )

    lineups, _ = _build_lineups(df, {}, None, quiet=True)
    assert len(lineups) == 1
    lineup = lineups[0]

    assert lineup.slots["C1"].player_id == "12345"
    assert _format_player_cell(lineup.slots["C1"]) == "Alpha (12345)"
    assert lineup.slots["W1"].player_id == "22222"
    assert _format_player_cell(lineup.slots["W1"]) == "Charlie (22222)"
    # Ensure NaN IDs remain absent
    assert lineup.slots["UTIL2"].player_id is None
    assert _format_player_cell(lineup.slots["UTIL2"]) == "Juliet"

    _prepare_sim(lineups, cfg={})

    expected_salary = 5200 + 5100 + 4900 + 4700 + 4500 + 4400 + 4300 + 7800 + 3600 + 3500
    expected_projection = sum(
        [9.5, 9.0, 8.6, 8.2, 7.5, 7.2, 6.9, 8.0, 5.1, 4.9]
    )

    assert math.isclose(lineup.metrics["salary"], expected_salary)
    assert math.isclose(lineup.metrics["projection"], expected_projection)
    assert math.isfinite(lineup.base_score)
