import re
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

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


def line_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize EV/PP fields and build stack tags for skaters."""

    out = df.copy()

    if "EV_Line" in out.columns:
        out["EV_Line"] = out["EV_Line"].map(_to_int_or_none)
    else:
        out["EV_Line"] = None

    if "PP_Unit" in out.columns:
        out["PP_Unit"] = out["PP_Unit"].map(_to_int_or_none)
    else:
        out["PP_Unit"] = None

    if "PosCanon" not in out.columns:
        out["PosCanon"] = None
    is_skater = out["PosCanon"].isin(["C", "W", "D"])
    out["IsSkater"] = is_skater.astype(bool)

    def _evrow(r):
        if not is_skater.iloc[r.name]:
            return None
        if pd.isna(r.get("Team")) or pd.isna(r.get("EV_Line")):
            return None
        return f"{r['Team']}-L{int(r['EV_Line'])}"

    def _pprow(r):
        if not is_skater.iloc[r.name]:
            return None
        if pd.isna(r.get("Team")) or pd.isna(r.get("PP_Unit")):
            return None
        return f"{r['Team']}-PP{int(r['PP_Unit'])}"

    out["EV_TAG"] = out.apply(_evrow, axis=1)
    # legacy column name expected by optimizer/output code
    out["EV_LINE_TAG"] = out["EV_TAG"]
    out["PP_TAG"] = out.apply(_pprow, axis=1)
    return out


def _collect_members(df: pd.DataFrame, tag_col: str, min_size: int) -> Dict[str, List[str]]:
    if tag_col not in df.columns:
        return {}

    tagged = df.dropna(subset=[tag_col])
    if "IsSkater" in tagged.columns:
        tagged = tagged[tagged["IsSkater"].astype(bool)]

    groups: Dict[str, List[str]] = {}
    for tag, grp in tagged.groupby(tag_col):
        members = grp["Name"].astype(str).tolist()
        if len(members) >= min_size:
            groups[str(tag)] = members
    return groups


def group_line_members(
    df: pd.DataFrame,
    tags: Iterable[str] = ("EV_TAG", "PP_TAG"),
    *,
    min_size: int = 2,
) -> Dict[str, Dict[str, List[str]]]:
    """Return line/PP groupings keyed by tag column."""

    out: Dict[str, Dict[str, List[str]]] = {}
    for col in tags:
        members = _collect_members(df, col, min_size)
        if members:
            out[col] = members
    return out


def game_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return sorted team matchups present in the player pool."""

    if "Team" not in df.columns or "Opp" not in df.columns:
        return []

    games: Dict[FrozenSet[str], Tuple[str, str]] = {}

    for _, row in df.iterrows():
        team = row.get("Team")
        opp = row.get("Opp")
        if not isinstance(team, str) or not isinstance(opp, str):
            continue
        team = team.strip().upper()
        opp = opp.strip().upper()
        if not team or not opp or team == opp:
            continue
        key = frozenset((team, opp))
        games.setdefault(key, (team, opp))

    # sort deterministically by tuple order
    ordered = sorted(games.values(), key=lambda t: (t[0], t[1]))
    return ordered
