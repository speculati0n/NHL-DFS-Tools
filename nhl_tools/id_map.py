from __future__ import annotations
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


LINEUP_SLOT_PREFIXES = ("C", "W", "D", "G", "UTIL")
DEFAULT_SLOT_ORDER = ("C", "W", "D", "G")  # used for UTIL and fallbacks


def _clean_name(s: str) -> str:
    """Strip trailing '(...)' and excess whitespace."""
    if not isinstance(s, str):
        return s
    s2 = re.sub(r"\s*\([^)]*\)\s*$", "", s)  # remove trailing "(...)"
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2


def _split_positions(val) -> List[str]:
    """Normalize 'C/W', 'C, W', 'C+W' → ['C','W'] (uppercased)."""
    if pd.isna(val):
        return []
    s = str(val).upper()
    s = re.sub(r"[^\w/,+ ]", "", s)
    s = s.replace(",", "/").replace("+", "/")
    parts = [p.strip() for p in s.split("/") if p.strip()]
    return parts


def _intended_pos_from_col(colname: str) -> str:
    u = str(colname).upper()
    if u.startswith("UTIL"):
        return "UTIL"
    for p in ("C", "W", "D", "G"):
        if u.startswith(p):
            return p
    return "UTIL"


def _detect_lineup_columns(sim_df: pd.DataFrame, player_name_set: Set[str]) -> List[str]:
    """Heuristic: object columns where >50% of cleaned values are known names,
    then filter by slot-like prefixes. Falls back to all candidates if no prefix match."""
    candidate_cols: List[str] = []
    for col in sim_df.columns:
        if sim_df[col].dtype == "object":
            series = sim_df[col].astype(str).map(_clean_name)
            match_ratio = (series.isin(player_name_set)).mean()
            if match_ratio > 0.5:
                candidate_cols.append(col)

    slot_cols: List[str] = []
    for col in candidate_cols:
        cname = str(col).upper()
        if cname.startswith(LINEUP_SLOT_PREFIXES) or re.match(r"^(C|W|D|G|UTIL)\d*$", cname):
            slot_cols.append(col)

    if not slot_cols:
        slot_cols = candidate_cols
    return slot_cols


def build_player_maps(
    ids_df: pd.DataFrame,
    name_col: str = "Name",
    id_col: str = "ID",
    pos_col: str = "Roster Position",
):
    """Builds:
    - lookup_by_name_pos[(clean_name, POS)] → ID
    - name_to_unique_id[clean_name] → ID (only if unique in file)
    - player_name_set
    """
    ids_df = ids_df.copy()
    # Normalize columns (case-insensitive support)
    cols = {c.lower(): c for c in ids_df.columns}
    name_col = cols.get(name_col.lower(), cols.get("name", name_col))
    id_col = cols.get(id_col.lower(), cols.get("id", id_col))
    pos_col = cols.get(pos_col.lower(), cols.get("roster_position", pos_col))

    ids_df["_clean_name"] = ids_df[name_col].astype(str).map(_clean_name)
    ids_df["_pos_list"] = ids_df[pos_col].map(_split_positions)

    lookup_by_name_pos: Dict[Tuple[str, str], str] = {}
    name_to_unique_id: Dict[str, str] = {}

    for nm, group in ids_df.groupby("_clean_name"):
        unique_ids = group[id_col].astype(str).unique().tolist()
        if len(unique_ids) == 1:
            name_to_unique_id[nm] = unique_ids[0]
        for _, row in group.iterrows():
            pid = str(row[id_col])
            for p in row["_pos_list"] or []:
                key = (nm, p.upper())
                # first occurrence wins; consistent IDs expected
                lookup_by_name_pos.setdefault(key, pid)

    player_name_set: Set[str] = set(ids_df["_clean_name"].unique())
    return lookup_by_name_pos, name_to_unique_id, player_name_set


def map_name_to_name_id(
    name_value: str,
    slot_col: str,
    lookup_by_name_pos: Dict[Tuple[str, str], str],
    name_to_unique_id: Dict[str, str],
    player_name_set: Set[str],
) -> str:
    """Map a single cell to 'Name (ID)'. If unresolved, return cleaned name."""
    nm = _clean_name(name_value) if isinstance(name_value, str) else name_value
    if not isinstance(nm, str) or nm == "" or nm not in player_name_set:
        return name_value  # leave non-player strings untouched

    desired = _intended_pos_from_col(slot_col)
    pid: Optional[str] = None

    if desired == "UTIL":
        for p in DEFAULT_SLOT_ORDER:
            pid = lookup_by_name_pos.get((nm, p))
            if pid:
                break
        if not pid:
            pid = name_to_unique_id.get(nm)
    else:
        pid = lookup_by_name_pos.get((nm, desired))
        if not pid:
            for p in DEFAULT_SLOT_ORDER:
                pid = lookup_by_name_pos.get((nm, p))
                if pid:
                    break
            if not pid:
                pid = name_to_unique_id.get(nm)

    if not pid:
        return nm
    return f"{nm} ({pid})"


def apply_name_id_mapping(
    sim_df: pd.DataFrame,
    ids_df: pd.DataFrame,
    explicit_slots: Optional[Iterable[str]] = None,
    name_col: str = "Name",
    id_col: str = "ID",
    pos_col: str = "Roster Position",
    log_prefix: str = "[id_map]",
) -> pd.DataFrame:
    """Return a copy of sim_df with lineup columns rewritten as 'Name (ID)'."""
    lookup_by_name_pos, name_to_unique_id, player_name_set = build_player_maps(
        ids_df, name_col=name_col, id_col=id_col, pos_col=pos_col
    )

    slot_cols = list(explicit_slots) if explicit_slots else _detect_lineup_columns(sim_df, player_name_set)
    out = sim_df.copy()

    # Remap and collect unresolved
    unresolved: Set[str] = set()
    changed = 0
    for col in slot_cols:
        def _mapper(x):
            before = x
            after = map_name_to_name_id(x, col, lookup_by_name_pos, name_to_unique_id, player_name_set)
            if isinstance(before, str):
                clean_before = _clean_name(before)
                if clean_before not in player_name_set:
                    return before
                # unresolved if unchanged but we cleaned parentheses
                if after == clean_before:
                    unresolved.add(clean_before)
                elif after != before:
                    nonlocal_changed[0] += 1
            return after

        nonlocal_changed = [0]
        out[col] = out[col].map(_mapper)
        changed += nonlocal_changed[0]

    # Simple log to stdout (caller can hook their logger instead)
    print(f"{log_prefix} remapped_cells={changed} slots={slot_cols}")
    if unresolved:
        print(f"{log_prefix} unresolved_names={sorted(unresolved)[:50]}{' ...' if len(unresolved)>50 else ''}")
    return out
