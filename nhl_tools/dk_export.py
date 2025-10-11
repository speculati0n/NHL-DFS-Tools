#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NHL → DraftKings-style wide export, mirroring NFL writer behavior:
- Columns are roster slots (positions as columns).
- Cells are "Player Name (playerid)".
- Reads NHL player IDs from dk_data/player_ids.csv (or user-provided path).
- Accepts either "tall" (LineupID, Slot, Name, ...) or "wide" internal lineups and normalizes.
"""

from __future__ import annotations
import os, re, unicodedata
from typing import Dict, List, Optional, Tuple
import pandas as pd

# DraftKings NHL Classic is 9 slots:
# 2C, 3W, 2D, 1G, 1UTIL  → ["C1","C2","W1","W2","W3","D1","D2","G","UTIL"]
DK_NHL_SLOTS = ["C1","C2","W1","W2","W3","D1","D2","G","UTIL"]

_PUNCT = r"[^\w\s]"

def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(_PUNCT, " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # drop suffixes
    s = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick(df: pd.DataFrame, *cands: str) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in cols:
            return cols[c.lower()]
    return None

def _first_initial_key(nm: str) -> Optional[str]:
    parts = [p for p in nm.split(" ") if p]
    if len(parts) < 2:
        return None
    first, rest = parts[0], parts[1:]
    if not first:
        return None
    return f"{first[0]} {' '.join(rest)}".strip()


def _build_id_index(ids_df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    name_col = _pick(ids_df, "name", "player", "full_name", "playername", "dk_name")
    id_col   = _pick(ids_df, "player_id", "playerid", "dk_id", "draftkings_id", "id")
    if not name_col or not id_col:
        raise ValueError("player_ids.csv must have a name column and a dk id column.")
    idx: Dict[str, Tuple[str, str]] = {}
    initial_candidates: Dict[str, List[Tuple[str, str]]] = {}
    for _, r in ids_df.iterrows():
        nm = _normalize_name(r[name_col])
        pid = str(r[id_col]).strip()
        canon = str(r[name_col]).strip()
        if nm and pid and canon:
            idx[nm] = (canon, pid)
            alt = _first_initial_key(nm)
            if alt and alt not in idx:
                initial_candidates.setdefault(alt, []).append((canon, pid))

    for alt, pids in initial_candidates.items():
        uniq = {pid for _, pid in pids}
        if len(uniq) == 1 and alt not in idx:
            idx[alt] = pids[0]
    return idx

def _fmt_cell(name: str, pid: Optional[str]) -> str:
    if not name:
        return ""
    pid = pid or ""
    return f"{name} ({pid})"

def _detect_wide_slots(df: pd.DataFrame) -> Optional[List[str]]:
    # If the DataFrame already has DK slots, return them in order.
    present = [c for c in DK_NHL_SLOTS if c in df.columns]
    if len(present) == len(DK_NHL_SLOTS):
        return DK_NHL_SLOTS
    # Alternate internal naming (e.g., C1_name etc.)
    present_split = [f"{c}_name" for c in DK_NHL_SLOTS]
    if all(c in df.columns for c in present_split):
        return present_split
    return None

def _tall_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tall → wide.
    Expects columns: LineupID, Slot, Name  (+ anything else ignored).
    """
    lineup_col = _pick(df, "LineupID", "lineup_id", "lineup", "k", "id")
    slot_col   = _pick(df, "Slot", "slot", "Pos", "Position")
    name_col   = _pick(df, "Name", "name", "Player")
    if not (lineup_col and slot_col and name_col):
        raise ValueError("Tall lineups require columns: LineupID, Slot, Name")

    # Normalized slot names like "C1","W2",...
    def _norm_slot(x: str) -> str:
        s = str(x).strip().upper()
        # common inputs: C, C1, W3, D2, G, UTIL
        m = re.match(r"^(C|W|D|G|UTIL)(\d+)?$", s)
        if m:
            p = m.group(1)
            n = m.group(2)
            if p == "C":
                return f"C{n or '1'}"
            if p == "W":
                return f"W{n or '1'}"
            if p == "D":
                return f"D{n or '1'}"
            return p
        return s

    tmp = df[[lineup_col, slot_col, name_col]].copy()
    tmp.columns = ["LineupID", "Slot", "Name"]
    tmp["Slot"] = tmp["Slot"].map(_norm_slot)

    # Pivot into columns; extra slots (like W4) are dropped; missing filled as empty
    wide = tmp.pivot_table(index="LineupID", columns="Slot", values="Name", aggfunc="first").reset_index()
    # Ensure required columns present in order
    out = pd.DataFrame()
    out["LineupID"] = wide["LineupID"]
    for c in DK_NHL_SLOTS:
        out[c] = wide[c] if c in wide.columns else ""
    return out

def parse_lineups_any(path: str) -> pd.DataFrame:
    """
    Read internal lineup CSV and return wide df with DK_NHL_SLOTS columns + LineupID.
    Supports:
      1) Already-wide with DK_NHL_SLOTS
      2) Wide with *_name columns
      3) Tall (LineupID, Slot, Name)
    """
    df = pd.read_csv(path)
    slots = _detect_wide_slots(df)
    if slots == DK_NHL_SLOTS:
        # Already in final shape (maybe without LineupID); add if missing
        if "LineupID" not in df.columns:
            df = df.copy()
            df.insert(0, "LineupID", range(1, len(df) + 1))
        return df[["LineupID"] + DK_NHL_SLOTS]
    if slots is not None:
        # Collapse *_name into slot
        out = pd.DataFrame({"LineupID": range(1, len(df) + 1)})
        for c in DK_NHL_SLOTS:
            out[c] = df[f"{c}_name"]
        return out
    # Try tall
    return _tall_to_wide(df)

def format_with_ids(wide_df: pd.DataFrame, player_ids_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map names to ids and return cells as "Name (playerid)" for DK_NHL_SLOTS.
    """
    idx = _build_id_index(player_ids_df)
    out = wide_df.copy()
    unmatched: List[Tuple[int,str,str]] = []  # (LineupID, Slot, Name)

    for c in DK_NHL_SLOTS:
        vals = []
        for i, name in enumerate(out[c].fillna("").astype(str).tolist()):
            nm = _normalize_name(name)
            match = idx.get(nm)
            if match:
                canon_name, pid = match
                vals.append(_fmt_cell(canon_name, pid))
            else:
                pid = ""
                if name:
                    unmatched.append((int(out.loc[i, "LineupID"]), c, name))
                    vals.append(_fmt_cell(name, pid))
                else:
                    vals.append("")
        out[c] = vals

    # Attach unmatched list for sidecar writing
    out.__dict__["_unmatched"] = unmatched
    return out

def export(lineups_path: str, ids_path: str, out_path: str, strict: bool = False) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    wide = parse_lineups_any(lineups_path)
    ids_df = pd.read_csv(ids_path)
    final = format_with_ids(wide, ids_df)
    final.to_csv(out_path, index=False)
    unmatched = getattr(final, "_unmatched", [])
    if strict and unmatched:
        raise SystemExit(f"Unmatched players: {len(unmatched)} (see sidecar).")
    if unmatched:
        side = re.sub(r"\.csv$", "", out_path) + "_unmatched.csv"
        pd.DataFrame(unmatched, columns=["LineupID","Slot","Name"]).to_csv(side, index=False)
