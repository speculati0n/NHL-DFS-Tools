import numpy as np

from nhl_tools.nhl_simulator import (
    DK_SLOTS,
    Lineup,
    PlayerRecord,
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
