# nhl_tools/nhl_projection_loader.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations


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


    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


        if need not in df.columns:
            raise ValueError(f"{path}: missing required column '{need}' after rename")
    # type cleanup
    df["salary"] = (

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


    for p in POS_ORDER:
        if overrides and overrides.get(p):
            paths[p] = overrides[p]
            continue

        # pick most recent for that pos
        paths[p] = sorted(g)[-1]
    return paths


    paths = resolve_paths(labs_dir, date, explicit)
    frames: List[pd.DataFrame] = []
    for pos, path in paths.items():
        frames.append(_read_one(path, pos))
    full = pd.concat(frames, ignore_index=True)
    # de-dup if same player shows up (rare), favor higher proj_points

