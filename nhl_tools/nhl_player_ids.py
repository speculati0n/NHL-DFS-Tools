# nhl_tools/nhl_player_ids.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import logging
import re
import unicodedata
from typing import Dict


log = logging.getLogger(__name__)


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # drop suffixes
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\. ?", "", s).strip()
    return s


def _key(name: str, team: str, pos: str) -> str:
    return f"{_norm_name(name)}|{(team or '').strip().upper()}|{(pos or '').strip().upper()}"


def load_player_ids(path: str) -> Dict[str, str]:
    mp: Dict[str, str] = {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                pid = str(row.get("player_id", "")).strip()
                name = row.get("name", "")
                team = row.get("team", "")
                pos = row.get("pos", "")
                if pid:
                    mp[_key(name, team, pos)] = pid
    except FileNotFoundError:
        log.warning("Player ID mapping file missing at %s", path)
    return mp


def lookup_player_id(name: str, team: str, pos: str, mp: Dict[str, str]) -> str:
    if not mp:
        return ""
    key = _key(name, team, pos)
    if key in mp:
        return mp[key]
    # try without team as fallback
    fallback = _key(name, "", pos)
    if fallback in mp:
        return mp[fallback]
    return ""


def decorate_with_ids(name: str, team: str, pos: str, mp: Dict[str, str]) -> str:
    pid = lookup_player_id(name, team, pos, mp)
    if not pid:
        log.warning("Missing player_id mapping for %s (%s %s)", name, team, pos)
    return f"{name} ({pid})"

