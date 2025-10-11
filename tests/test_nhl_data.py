import numpy as np
import pandas as pd

from nhl_tools.nhl_data import normalize_lineup_player, normalize_name


def _player_row(pp_unit_value):
    return pd.Series(
        {
            "Name": "John Doe",
            "Team": "AAA",
            "Pos": "C",
            "Opp": "BBB",
            "Salary": 5000,
            "Proj": 10.0,
            "Ceiling": 12.0,
            "Actual": 0.0,
            "Ownership": 5.0,
            "Full": "1F",
            "pp_unit": pp_unit_value,
            "PlayerID": "123",
        }
    )


def test_normalize_lineup_player_handles_missing_pp_unit():
    meta = {"team": "AAA", "pos": "C"}

    lookup_with_nan = {normalize_name("John Doe"): [_player_row(np.nan)]}
    result_nan = normalize_lineup_player("John Doe", meta, lookup_with_nan)
    assert result_nan["pp_unit"] is None

    lookup_with_empty = {normalize_name("John Doe"): [_player_row("")]}
    result_empty = normalize_lineup_player("John Doe", meta, lookup_with_empty)
    assert result_empty["pp_unit"] is None
