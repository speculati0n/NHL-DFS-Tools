import re
from typing import Optional

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _to_int_or_none(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    try:
        return int(float(s))
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def _clean_team(val: Optional[str]) -> Optional[str]:
    """Return an uppercase team code if ``val`` looks valid."""

    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    team = str(val).strip().upper()
    if not team or team in {"", "NONE", "NAN"}:
        return None
    return team


def line_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stack-friendly tags for EV lines and power-play units.

    The optimizer expects two helper columns on the projection set:

    ``EV_LINE_TAG``
        Normalised identifier for even-strength lines (forwards or defence).
        Goalies are excluded.  Values look like ``BOS_L1`` or ``COL_D2``.

    ``PP_TAG``
        Identifier for power-play units such as ``FLA_PP1``.

    FantasyLabs' CSV exports occasionally omit or reformat the line columns, so
    we defensively parse the values here before assigning tags.  Any skaters
    without usable information keep ``None`` in the new columns which naturally
    drops them from stack constraints later on.
    """

    out = df.copy()

    def make_ev_tag(row) -> Optional[str]:
        if str(row.get("PosCanon", "")).upper() == "G":
            return None
        team = _clean_team(row.get("Team"))
        if not team:
            return None

        # Prefer the parsed integer from EV_Line; fall back to the raw string
        # when necessary (e.g., "1F", "2 D").
        line_no = row.get("EV_Line")
        line_id = _to_int_or_none(line_no)
        if line_id is None:
            line_id = _to_int_or_none(row.get("FullRaw"))
        if line_id is None:
            return None

        ev_type = str(row.get("EV_Type", "")).strip().upper() or "F"
        if ev_type == "D":
            suffix = f"D{line_id}"
        else:
            suffix = f"L{line_id}"
        return f"{team}_{suffix}"

    def make_pp_tag(row) -> Optional[str]:
        if str(row.get("PosCanon", "")).upper() == "G":
            return None
        team = _clean_team(row.get("Team"))
        if not team:
            return None
        unit = row.get("PP_Unit")
        unit_id = _to_int_or_none(unit)
        if unit_id is None:
            unit_id = _to_int_or_none(row.get("PP"))
        if unit_id is None:
            return None
        return f"{team}_PP{unit_id}"

    out["EV_LINE_TAG"] = out.apply(make_ev_tag, axis=1)
    out["PP_TAG"] = out.apply(make_pp_tag, axis=1)
    return out


def group_line_members(df: pd.DataFrame, tag_col: str) -> dict[str, list[str]]:
    """Return mapping of line tag -> player names (skip missing tags)."""

    if tag_col not in df.columns:
        return {}

    groups: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        tag = row.get(tag_col)
        if not tag or (isinstance(tag, float) and pd.isna(tag)):
            continue
        groups.setdefault(str(tag), []).append(str(row.get("Name", "")))
    return groups


def game_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return sorted team matchups present in *df*.

    The input frame already carries ``Team``/``Opp`` columns from the Labs
    export.  We normalise those strings and create a stable, deduplicated list of
    pairs such as ``[("BOS", "NYR"), ...]`` so the optimizer can set bring-back
    constraints on a per-game basis.
    """

    seen = set()
    for _, row in df.iterrows():
        team = _clean_team(row.get("Team"))
        opp = _clean_team(row.get("Opp"))
        if not team or not opp:
            continue
        if team == opp:
            continue
        pair = tuple(sorted((team, opp)))
        seen.add(pair)
    return sorted(seen)
