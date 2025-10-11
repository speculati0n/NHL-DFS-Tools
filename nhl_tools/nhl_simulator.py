#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .nhl_projection_loader import load_labs_for_date as load_labs_raw
from nhl_tools.dk_export import DK_NHL_SLOTS, parse_lineups_any as _parse_lineups_any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_LABS_DIR = os.path.join(REPO_ROOT, "dk_data")

log = logging.getLogger("nhl_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _sanitize_date_for_loader(date: str) -> str:
    return date.replace("-", "")


def _prepare_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Name"] = out["name"].astype(str).str.strip()
    out["Team"] = out["team"].astype(str).str.upper().str.strip()
    out["PosCanon"] = out["pos"].astype(str).str.upper()
    out["Proj"] = out["proj_points"].astype(float)
    return out


def _sample_points(mean: float, vol: float, rng: np.random.Generator) -> float:
    mean = max(mean, 0.1)
    sigma = max(0.15 * mean, vol * mean)
    mu = math.log(mean**2 / math.sqrt(sigma**2 + mean**2))
    s = math.sqrt(math.log(1 + (sigma**2) / (mean**2)))
    return float(lognorm(s=s, scale=math.exp(mu)).rvs(random_state=rng))


def _load_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    log.info("Loading config from %s", path)
    if path.lower().endswith((".yaml", ".yml")):
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    overrides = {"C": args.c_file, "W": args.w_file, "D": args.d_file, "G": args.g_file}
    return {k: v for k, v in overrides.items() if v}


def simulate_lineups(
    lineups_csv: str,
    labs_dir: str,
    date: str,
    sims: int,
    ceil_mult: float,
    floor_mult: float,
    w_up: float,
    w_con: float,
    w_dud: float,
    seed: int = 42,
    overrides: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    raw = load_labs_raw(labs_dir, _sanitize_date_for_loader(date), overrides if overrides else None)
    base = _prepare_pool(raw)
    base_lookup = base.copy()
    base_lookup["NameNorm"] = base_lookup["Name"].astype(str).str.lower()
    base_lookup["PosCanonNorm"] = base_lookup["PosCanon"].astype(str).str.upper()

    wide = _parse_lineups_any(lineups_csv)
    slot_cols = [c for c in DK_NHL_SLOTS if c in wide.columns]
    if "LineupID" not in wide.columns:
        wide.insert(0, "LineupID", range(1, len(wide) + 1))
    lineup_names = wide[["LineupID"] + slot_cols].copy()

    slot_pos = {
        "C1": "C",
        "C2": "C",
        "W1": "W",
        "W2": "W",
        "W3": "W",
        "D1": "D",
        "D2": "D",
        "G": "G",
        "UTIL": None,
    }

    records: List[Dict[str, object]] = []
    for _, row in lineup_names.iterrows():
        lid = int(row["LineupID"])
        for slot in slot_cols:
            raw_name = row[slot]
            if pd.isna(raw_name):
                continue
            name_str = str(raw_name).strip()
            if not name_str:
                continue
            clean_name = re.sub(r"\s*\([^)]*\)\s*$", "", name_str).strip()
            pos_hint = slot_pos.get(slot)
            candidates = base_lookup[base_lookup["NameNorm"] == clean_name.lower()]
            if pos_hint:
                candidates = candidates[candidates["PosCanonNorm"] == pos_hint]
            if candidates.empty:
                team = None
                pos = pos_hint
                proj = np.nan
            else:
                picked = candidates.iloc[0]
                team = picked["Team"]
                pos = picked["PosCanon"]
                proj = float(picked["Proj"])
            records.append(
                {
                    "LineupID": lid,
                    "Slot": slot,
                    "Name": clean_name,
                    "Team": team,
                    "PosCanon": pos,
                    "Proj": proj,
                }
            )

    lu = pd.DataFrame(records)
    if lu.empty:
        raise ValueError("No players parsed from lineups CSV.")

    lu["Mean"] = lu["Proj"]
    if lu["Mean"].isna().any():
        log.warning("Some players missing from Labs projections; dropping those entries for simulation.")
        lu = lu.dropna(subset=["Mean"])

    lu["PosCanon"] = lu["PosCanon"].astype(str)
    lu["Ceil"] = lu["Mean"] * ceil_mult
    lu["Floor"] = lu["Mean"] * floor_mult
    g = lu.groupby("PosCanon")
    upZ = (lu["Ceil"] - g["Ceil"].transform("mean")) / (g["Ceil"].transform("std").replace(0, np.nan))
    coZ = (lu["Floor"] - g["Floor"].transform("mean")) / (g["Floor"].transform("std").replace(0, np.nan))
    lu["Vol"] = (0.30 + 0.12 * w_up * upZ.abs().fillna(0) - 0.05 * w_con * coZ.clip(upper=0).abs().fillna(0)).clip(0.15, 0.6)

    rng = np.random.default_rng(seed)
    results = {lid: [] for lid in sorted(lu["LineupID"].unique())}
    for _ in range(int(sims)):
        sampled = lu.copy()
        sampled["Pts"] = sampled.apply(lambda r: _sample_points(r["Mean"], r["Vol"], rng), axis=1)
        totals = sampled.groupby("LineupID")["Pts"].sum()
        for lid, val in totals.items():
            results[lid].append(float(val))

    rows = []
    for lid, arr in results.items():
        a = np.array(arr)
        rows.append(
            {
                "LineupID": lid,
                "SimMean": float(a.mean()),
                "P90": float(np.percentile(a, 90)),
                "P75": float(np.percentile(a, 75)),
                "P50": float(np.percentile(a, 50)),
                "P25": float(np.percentile(a, 25)),
                "Std": float(a.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("P90", ascending=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("NHL lineup simulator (NFL-style CLI)")
    ap.add_argument("--lineups", required=True, help="CSV of lineups (optimizer output)")
    ap.add_argument("--labs-dir", default=DEFAULT_LABS_DIR, help="Directory of FantasyLabs NHL CSVs")
    ap.add_argument("--date", required=True, help="Slate date (YYYY-MM-DD)")
    ap.add_argument("--sims", type=int, default=10000, help="Number of tournament simulations")
    ap.add_argument("--out", required=True, help="Output CSV for simulation summary")
    ap.add_argument("--config", type=str, help="Optional YAML/JSON config for weights/overrides")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    ap.add_argument("--contest", type=str, help="Contest file (JSON/CSV) for parity with NFL sims")
    ap.add_argument("--c-file", type=str, help="Explicit C projections file")
    ap.add_argument("--w-file", type=str, help="Explicit W projections file")
    ap.add_argument("--d-file", type=str, help="Explicit D projections file")
    ap.add_argument("--g-file", type=str, help="Explicit G projections file")

    ap.add_argument("--w-up", type=float, default=0.15)
    ap.add_argument("--w-con", type=float, default=0.05)
    ap.add_argument("--w-dud", type=float, default=0.03)
    ap.add_argument("--ceil-mult", type=float, default=1.6)
    ap.add_argument("--floor-mult", type=float, default=0.55)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> str:
    args = parse_args(argv)
    cfg = _load_config(args.config)
    overrides = _resolve_paths(args)

    weights_cfg = cfg.get("weights") or {}
    w_up = float(weights_cfg.get("upside", args.w_up))
    w_con = float(weights_cfg.get("consistency", args.w_con))
    w_dud = float(weights_cfg.get("duds", args.w_dud))
    ceil_mult = float(cfg.get("ceil_mult", args.ceil_mult))
    floor_mult = float(cfg.get("floor_mult", args.floor_mult))

    if args.contest and not cfg.get("contest"):
        log.warning("Contest files are not yet modeled in the NHL simulator; flag accepted for CLI parity only.")

    report = simulate_lineups(
        args.lineups,
        args.labs_dir,
        args.date,
        args.sims,
        ceil_mult,
        floor_mult,
        w_up,
        w_con,
        w_dud,
        seed=int(cfg.get("seed", args.seed)),
        overrides=overrides,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    report.to_csv(args.out, index=False)
    log.info("Wrote %s", args.out)
    return args.out


if __name__ == "__main__":
    main()
