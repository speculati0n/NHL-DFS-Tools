"""DraftKings upload helpers for the edges optimizer."""
from __future__ import annotations

import os
import unicodedata
import re
from typing import Dict, List, Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from nhl_tools.nhl_optimizer import LineupResult

UPLOAD_COLUMNS = ["C", "W", "W", "D", "D", "G", "UTIL", "UTIL"]


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_id_index(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    name_col = None
    for cand in ["name", "player", "full_name", "playername", "dk_name"]:
        if cand in cols:
            name_col = cols[cand]
            break
    id_col = None
    for cand in ["player_id", "playerid", "dk_id", "draftkings_id", "id"]:
        if cand in cols:
            id_col = cols[cand]
            break
    if not name_col or not id_col:
        raise ValueError("player_ids.csv missing name/id columns")
    idx: Dict[str, str] = {}
    for _, row in df.iterrows():
        nm = _normalize_name(row[name_col])
        pid = str(row[id_col]).strip()
        if nm and pid:
            idx[nm] = pid
    return idx


def _format_cell(name: str, pid: str) -> str:
    pid = pid or ""
    return f"{name} ({pid})"


def _map_ids(names: Sequence[str], idx: Dict[str, str]) -> List[str]:
    out = []
    for name in names:
        nm = _normalize_name(name)
        pid = idx.get(nm, "")
        out.append(_format_cell(name, pid))
    return out


def export_dk_upload(
    lineups: Sequence["LineupResult"],
    pool: pd.DataFrame,
    player_ids_path: str,
    out_path: str,
) -> None:
    """Write a DraftKings upload CSV for a sequence of lineups."""

    if not lineups:
        raise ValueError("No lineups provided")

    id_index = _build_id_index(player_ids_path)
    records: List[List[str]] = []

    for lu in lineups:
        slots = lu.slots
        c1 = slots.get("C1")
        c2 = slots.get("C2")
        wings = [slots.get("W1"), slots.get("W2"), slots.get("W3")]
        defenders = [slots.get("D1"), slots.get("D2")]
        goalie = slots.get("G")
        util = slots.get("UTIL")

        picked_order = []
        picked_order.append(c1)
        picked_order.append(wings[0])
        picked_order.append(wings[1])
        picked_order.append(defenders[0])
        picked_order.append(defenders[1])
        picked_order.append(goalie)
        picked_order.append(util)

        consumed = {idx for idx in picked_order if idx is not None}
        remaining = [i for i in lu.indices if i not in consumed]
        remaining.sort(key=lambda i: float(pool.loc[i, "Proj"]), reverse=True)
        picked_order.append(remaining[0] if remaining else None)

        names = [pool.loc[i, "Name"] if i is not None else "" for i in picked_order]
        records.append(_map_ids(names, id_index))

    out_df = pd.DataFrame(records, columns=UPLOAD_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)

