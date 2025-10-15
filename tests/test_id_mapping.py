import pandas as pd

from nhl_tools.dk_export import _build_id_index
from nhl_tools.id_mapping import find_pid, load_player_ids_any


def test_build_id_index_handles_abbreviated_names():
    df = pd.DataFrame(
        {
            "name": ["Cameron Talbot"],
            "player_id": ["123"],
        }
    )

    idx = _build_id_index(df)

    assert idx["cameron talbot"] == ("Cameron Talbot", "123")
    assert idx["c talbot"] == ("Cameron Talbot", "123")
    assert idx["cam talbot"] == ("Cameron Talbot", "123")


def test_build_id_index_skips_ambiguous_aliases():
    df = pd.DataFrame(
        {
            "name": ["Cameron Talbot", "Calvin Talbot"],
            "player_id": ["1", "2"],
        }
    )

    idx = _build_id_index(df)

    assert "c talbot" not in idx


def test_find_pid_matches_short_names(tmp_path):
    csv_path = tmp_path / "ids.csv"
    csv_path.write_text("player_id,name,team,pos\n123,Cameron Talbot,OTT,G\n", encoding="utf-8")

    mp = load_player_ids_any(str(csv_path))

    assert find_pid("Cameron Talbot", "OTT", "G", mp) == "123"
    assert find_pid("Cam Talbot", "OTT", "G", mp) == "123"
    assert find_pid("C. Talbot", "OTT", "G", mp) == "123"
