# nhl_tools/nhl_projection_loader.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, unicodedata, glob
import pandas as pd
from typing import Dict, List, Optional

POS_ORDER = ["C","W","D","G"]

# Expected FantasyLabs NHL per-position filename patterns
DEFAULT_PATTERNS = {
    "C": "fantasylabs_player_data_NHL_C_{date}.csv",
    "W": "fantasylabs_player_data_NHL_W_{date}.csv",
    "D": "fantasylabs_player_data_NHL_D_{date}.csv",
    "G": "fantasylabs_player_data_NHL_G_{date}.csv",
}

# Column mapping from Labs -> internal (adjust if your Labs headers differ)
COLMAP = {
    "Player": "name",
    "Team": "team",
    "Opp": "opp",
    "Salary": "salary",
    "Proj": "proj_points",
    "Own": "projected_own",
    # Optional flags often present in Labs
    "Full": "full_stack_flag",
    "PP": "powerplay_flag",
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def _read_one(path: str, pos: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # rename columns if present
    rename = {k:v for k,v in COLMAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    # ensure required
    for need in ["name","team","salary"]:
        if need not in df.columns:
            raise ValueError(f"{path}: missing required column '{need}' after rename")
    # type cleanup
    df["salary"] = (
        df["salary"].astype(str).str.replace(",","", regex=False).str.extract(r"(\d+)", expand=False).fillna("0").astype(int)
    )
    if "proj_points" not in df.columns:
        df["proj_points"] = 0.0
    if "projected_own" not in df.columns:
        df["projected_own"] = 0.0

    df["pos"] = pos
    # normalized keys for merge/mapping
    df["name_key"] = df["name"].map(_norm)
    df["team_key"] = df["team"].astype(str).str.upper().str.strip()
    if "opp" in df.columns:
        df["opp"] = df["opp"].astype(str).str.upper().str.strip()
    return df

def resolve_paths(labs_dir: str, date: str, overrides: Optional[Dict[str,str]]=None) -> Dict[str,str]:
    """Return file paths for C/W/D/G. If overrides provided, use them when specified."""
    paths: Dict[str,str] = {}
    for p in POS_ORDER:
        if overrides and overrides.get(p):
            paths[p] = overrides[p]
            continue
        pat = DEFAULT_PATTERNS[p].format(date=date)
        guess = os.path.join(labs_dir, pat)
        if os.path.exists(guess):
            paths[p] = guess
            continue
        # last-resort glob
        g = glob.glob(os.path.join(labs_dir, f"fantasylabs_player_data_NHL_{p}_*.csv"))
        if not g:
            raise FileNotFoundError(f"Missing Labs file for {p}. Tried {guess}")
        # pick most recent for that pos
        paths[p] = sorted(g)[-1]
    return paths

def load_labs_for_date(labs_dir: str, date: str, explicit: Optional[Dict[str,str]]=None) -> pd.DataFrame:
    paths = resolve_paths(labs_dir, date, explicit)
    frames: List[pd.DataFrame] = []
    for pos, path in paths.items():
        frames.append(_read_one(path, pos))
    full = pd.concat(frames, ignore_index=True)
    # de-dup if same player shows up (rare), favor higher proj_points
    full = full.sort_values(["name_key","pos","proj_points"], ascending=[True, True, False])
    full = full.drop_duplicates(subset=["name_key","pos"], keep="first")
    return full.reset_index(drop=True)
