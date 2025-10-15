#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import re
import unicodedata
from typing import Dict, Iterable, List, Optional, Set, Tuple

# We build a mapping keyed by normalized (name, team, pos) -> player_id.
# Supports two input schemas:
#   A) Simple mapping:  player_id,name,team,pos
#   B) DraftKings export: Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame

def _safe_str(x) -> str:
    return "" if x is None else str(x)

def _norm_name(s: str) -> str:
    s = _safe_str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # drop suffixes
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s).strip()
    return s


def _split_first_rest(nm: str) -> Optional[Tuple[str, str]]:
    parts = [p for p in nm.split(" ") if p]
    if len(parts) < 2:
        return None
    first, rest = parts[0], parts[1:]
    remainder = " ".join(rest).strip()
    if not first or not remainder:
        return None
    return first, remainder


def _iter_name_aliases(nm: str) -> List[str]:
    parts = _split_first_rest(nm)
    if not parts:
        return []
    first, rest = parts
    aliases = [f"{first[0]} {rest}"]
    for length in range(3, len(first)):
        prefix = first[:length]
        if prefix != first:
            aliases.append(f"{prefix} {rest}")
    # Deduplicate while preserving order
    seen = set()
    uniq_aliases = []
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            uniq_aliases.append(alias)
    return uniq_aliases

def _key(name: str, team: str, pos: str) -> str:
    return f"{_norm_name(name)}|{_safe_str(team).strip().upper()}|{_safe_str(pos).strip().upper()}"

def _keys_for_multi_elig(name: str, team: str, raw_pos: str) -> Iterable[str]:
    """
    DK can list combined eligibility, e.g., 'C/W', 'W/D'.
    We emit one mapping key per single-letter NHL position in {C,W,D,G}.
    """
    team = _safe_str(team).strip().upper()
    raw = _safe_str(raw_pos).upper().replace(" ", "")
    parts = [p for p in re.split(r"[/,]", raw) if p]
    if not parts:
        # Fallback: scan for any position letter in raw string
        parts = [p for p in ["C","W","D","G"] if p in raw]
    seen = set()
    for p in parts or [""]:
        if p in {"C","W","D","G"} and p not in seen:
            seen.add(p)
            yield _key(name, team, p)

def _detect_schema(fieldnames: Iterable[str]) -> str:
    cols = {c.strip() for c in fieldnames if c}
    # Simple schema
    if {"player_id","name","team","pos"} <= cols:
        return "simple"
    # DK export schema
    if {"Name","ID","TeamAbbrev"} <= cols:
        # Position can be in 'Position' or 'Roster Position'
        if "Position" in cols or "Roster Position" in cols:
            return "dk"
    return "unknown"

def _load_simple(reader: csv.DictReader) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    canonical_entries: List[Tuple[str, str, str, str]] = []  # (name_norm, team, pos, pid)
    for row in reader:
        pid = _safe_str(row.get("player_id", "")).strip()
        name = _safe_str(row.get("name", "")).strip()
        team = _safe_str(row.get("team", "")).strip().upper()
        pos = _safe_str(row.get("pos", "")).strip().upper()
        if not (pid and name and team and pos):
            continue
        key = _key(name, team, pos)
        mp[key] = pid
        canonical_entries.append((_norm_name(name), team, pos, pid))

    alias_candidates: Dict[str, Set[str]] = {}
    for name_norm, team, pos, pid in canonical_entries:
        for alias in _iter_name_aliases(name_norm):
            alias_key = f"{alias}|{team}|{pos}"
            if alias_key not in mp:
                alias_candidates.setdefault(alias_key, set()).add(pid)

    for alias_key, pid_set in alias_candidates.items():
        if len(pid_set) == 1 and alias_key not in mp:
            mp[alias_key] = next(iter(pid_set))
    return mp

def _load_dk(reader: csv.DictReader) -> Dict[str,str]:
    """
    Build mapping from a DK export-like CSV.
    Expected minimum columns:
      - Name (player name)
      - ID   (DK player id)
      - TeamAbbrev (e.g., EDM, TOR)
      - Position or Roster Position (can be single or multi like 'C/W')
    """
    mp: Dict[str, str] = {}
    alias_entries: List[Tuple[str, str, str, str]] = []
    for row in reader:
        name = _safe_str(row.get("Name","")).strip()
        pid  = _safe_str(row.get("ID","")).strip()
        team = _safe_str(row.get("TeamAbbrev","")).strip().upper()
        rawp = _safe_str(row.get("Position", row.get("Roster Position",""))).strip().upper()

        if not (name and pid and team):
            # Skip incomplete rows
            continue

        name_norm = _norm_name(name)
        keys = list(_keys_for_multi_elig(name, team, rawp))
        if not keys:
            # If we couldn't parse a position, attempt to infer one letter from raw string
            for p in ["C","W","D","G"]:
                if p in rawp:
                    keys = [_key(name, team, p)]
                    break
            if not keys:
                # Last-resort: create a pos-agnostic key (rarely used; only for fallback at decoration time)
                keys = [_key(name, team, "")]

        for k in keys:
            # Prefer first occurrence (avoid overriding with duplicates)
            if k not in mp:
                mp[k] = pid
                parts = k.split("|")
                if len(parts) == 3:
                    _, team_part, pos_part = parts
                    alias_entries.append((name_norm, team_part, pos_part, pid))

    alias_candidates: Dict[str, Set[str]] = {}
    for name_norm, team, pos, pid in alias_entries:
        for alias in _iter_name_aliases(name_norm):
            alias_key = f"{alias}|{team}|{pos}"
            if alias_key not in mp:
                alias_candidates.setdefault(alias_key, set()).add(pid)

    for alias_key, pid_set in alias_candidates.items():
        if len(pid_set) == 1 and alias_key not in mp:
            mp[alias_key] = next(iter(pid_set))
    return mp

def load_player_ids_any(path: str) -> Dict[str,str]:
    """
    Load a player-id map from either the simple schema or the DK export schema.
    Returns a dict keyed by normalized 'name|TEAM|POS' -> 'player_id'.
    """
    mp: Dict[str,str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            schema = _detect_schema(reader.fieldnames or [])
            if schema == "simple":
                mp = _load_simple(reader)
            elif schema == "dk":
                mp = _load_dk(reader)
            else:
                # Try best-effort DK load (common in the wild)
                mp = _load_dk(reader)
    except FileNotFoundError:
        # Return empty; caller can still export names without IDs
        return {}
    return mp

def find_pid(name: str, team: str, pos: str, mp: Dict[str,str]) -> str:
    """
    Retrieve player_id with sensible fallbacks:
      1) exact (name, team, pos)
      2) (name, team, '')  [if DK row lacked clean pos]
      3) (name, '', pos)   [rare]
      4) (name, '', '')    [name-only]
    """
    team = _safe_str(team).strip().upper()
    pos  = _safe_str(pos).strip().upper()
    cand = [
        _key(name, team, pos),
        _key(name, team, ""),
        _key(name, "", pos),
        _key(name, "", ""),
    ]
    for k in cand:
        pid = mp.get(k, "")
        if pid:
            return pid
    return ""
