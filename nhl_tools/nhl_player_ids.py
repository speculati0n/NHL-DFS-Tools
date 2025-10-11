# nhl_tools/nhl_player_ids.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import csv, re, unicodedata
from typing import Dict

def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    # drop suffixes
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", s).strip()
    return s

def _key(name: str, team: str, pos: str) -> str:
    return f"{_norm_name(name)}|{(team or '').strip().upper()}|{(pos or '').strip().upper()}"

def load_player_ids(path: str) -> Dict[str,str]:
    mp: Dict[str,str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pid = str(row.get("player_id","" )).strip()
            name = row.get("name","")
            team = row.get("team","")
            pos  = row.get("pos","")
            if pid:
                mp[_key(name, team, pos)] = pid
    return mp

def decorate_with_ids(name: str, team: str, pos: str, mp: Dict[str,str]) -> str:
    pid = mp.get(_key(name, team, pos), "")
    return f"{name} ({pid})"
