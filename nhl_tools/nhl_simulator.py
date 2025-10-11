#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .nhl_projection_loader import load_labs_for_date
from .nhl_optimizer import prepare_player_pool


log = logging.getLogger("nhl_sim")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DK_SALARY_CAP = 50000
DK_ROSTER_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
SLOT_POS_HINT = {
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


def _sample_points(mean: float, vol: float, rng: np.random.Generator) -> float:
    mean = max(mean, 0.1)
    sigma = max(0.15 * mean, vol * mean)
    mu = math.log(mean**2 / math.sqrt(sigma**2 + mean**2))
    s = math.sqrt(math.log(1 + (sigma**2) / (mean**2)))
    return float(lognorm(s=s, scale=math.exp(mu)).rvs(random_state=rng))


def load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("NHL Simulator (NFL-style CLI)")
    ap.add_argument("--lineups", required=True, help="Optimizer output CSV")
    ap.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo runs")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    ap.add_argument("--labs-dir", type=str, default="dk_data", help="Directory of FantasyLabs CSVs")
    ap.add_argument("--date", type=str, help="Slate date (YYYY-MM-DD); inferred from file if omitted")
    ap.add_argument("--config", type=str, help="YAML/JSON config overriding simulator settings")
    ap.add_argument("--contest", type=str, help="Contest metadata file (JSON/CSV)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for simulations")

    # legacy weights (deprecated)
    ap.add_argument("--w-up", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--w-con", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--w-dud", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--ceil-mult", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--floor-mult", type=float, default=None, help=argparse.SUPPRESS)
    return ap.parse_args()


def merge_settings(args: argparse.Namespace, cfg: Dict) -> Dict[str, float]:
    settings: Dict[str, float] = {}

    def pick(key: str, legacy_flag: str, default: float) -> float:
        if key in cfg:
            return float(cfg[key])
        val = getattr(args, legacy_flag)
        if val is not None:
            log.warning("Flag --%s is deprecated. Prefer using --config.", legacy_flag.replace("_", "-"))
            return float(val)
        return default

    settings["w_up"] = pick("w_up", "w_up", 0.15)
    settings["w_con"] = pick("w_con", "w_con", 0.05)
    settings["w_dud"] = pick("w_dud", "w_dud", 0.03)
    settings["ceil_mult"] = pick("ceil_mult", "ceil_mult", 1.6)
    settings["floor_mult"] = pick("floor_mult", "floor_mult", 0.55)
    settings["seed"] = int(cfg.get("seed", args.seed))
    return settings


def resolve_date(args: argparse.Namespace, cfg: Dict) -> str:
    if args.date:
        return args.date
    if "date" in cfg:
        return str(cfg["date"])
    pattern = re.compile(r"(20\d{2}-\d{2}-\d{2})")
    m = pattern.search(os.path.basename(args.lineups))
    if m:
        return m.group(1)
    raise ValueError("Slate date required (--date or config['date']).")


def read_contest_meta(path: Optional[str]) -> None:
    if not path:
        return
    if not os.path.exists(path):
        log.warning("Contest metadata file %s not found", path)
        return
    try:
        if path.lower().endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)
        else:
            pd.read_csv(path)
    except Exception as exc:
        log.warning("Failed to parse contest metadata %s: %s", path, exc)


def parse_lineups(lineups_csv: str) -> pd.DataFrame:
    wide = pd.read_csv(lineups_csv)
    if "LineupID" not in wide.columns:
        wide.insert(0, "LineupID", np.arange(1, len(wide) + 1))
    slot_cols = [c for c in DK_ROSTER_SLOTS if c in wide.columns]
    records = []
    for _, row in wide.iterrows():
        lid = int(row["LineupID"])
        for slot in slot_cols:
            val = row.get(slot)
            if pd.isna(val):
                continue
            name_str = str(val).strip()
            if not name_str:
                continue
            clean = re.sub(r"\s*\([^)]*\)\s*$", "", name_str).strip()
            if not clean:
                continue
            records.append({"LineupID": lid, "Slot": slot, "Name": clean})
    return pd.DataFrame(records)


def attach_projection_rows(players: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    if players.empty:
        raise ValueError("No players parsed from lineups CSV.")
    lookup = pool.copy()
    lookup["NameNorm"] = lookup["Name"].astype(str).str.lower()
    lookup["PosCanonNorm"] = lookup["PosCanon"].astype(str).str.upper()

    enriched = []
    for _, row in players.iterrows():
        name = row["Name"]
        slot = row["Slot"]
        pos_hint = SLOT_POS_HINT.get(slot)
        matches = lookup[lookup["NameNorm"] == name.lower()]
        if pos_hint:
            matches = matches[matches["PosCanonNorm"] == pos_hint]
        if matches.empty and pos_hint is None:
            matches = lookup[lookup["NameNorm"] == name.lower()]
        if matches.empty:
            log.warning("Player %s (%s) missing from projections", name, slot)
            continue
        picked = matches.iloc[0]
        enriched.append(
            {
                "LineupID": row["LineupID"],
                "Slot": slot,
                "Name": name,
                "Team": picked.get("Team"),
                "PosCanon": picked.get("PosCanon"),
                "Proj": float(picked.get("ProjBase", picked.get("Proj", 0.0))),
            }
        )
    return pd.DataFrame(enriched)


def simulate_lineups(lineups_csv: str, pool: pd.DataFrame, sims: int, settings: Dict[str, float]) -> pd.DataFrame:
    parsed = parse_lineups(lineups_csv)
    enriched = attach_projection_rows(parsed, pool)
    if enriched.empty:
        raise ValueError("No players matched projections for simulation.")

    enriched["Mean"] = enriched["Proj"].astype(float)
    ceil_mult = settings["ceil_mult"]
    floor_mult = settings["floor_mult"]
    enriched["Ceil"] = enriched["Mean"] * ceil_mult
    enriched["Floor"] = enriched["Mean"] * floor_mult
    g = enriched.groupby("PosCanon")
    upZ = (enriched["Ceil"] - g["Ceil"].transform("mean")) / (g["Ceil"].transform("std").replace(0, np.nan))
    coZ = (enriched["Floor"] - g["Floor"].transform("mean")) / (g["Floor"].transform("std").replace(0, np.nan))
    enriched["Vol"] = (
        0.30
        + 0.12 * settings["w_up"] * upZ.abs().fillna(0)
        - 0.05 * settings["w_con"] * coZ.clip(upper=0).abs().fillna(0)
    ).clip(0.15, 0.6)

    rng = np.random.default_rng(int(settings["seed"]))
    results = {lid: [] for lid in sorted(enriched["LineupID"].unique())}
    for _ in range(int(sims)):
        sampled = enriched.copy()
        sampled["Pts"] = sampled.apply(lambda r: _sample_points(r["Mean"], r["Vol"], rng), axis=1)
        totals = sampled.groupby("LineupID")["Pts"].sum()
        for lid, val in totals.items():
            results[lid].append(float(val))

    rows = []
    for lid, arr in results.items():
        arr_np = np.array(arr)
        rows.append(
            {
                "LineupID": lid,
                "SimMean": float(arr_np.mean()),
                "P90": float(np.percentile(arr_np, 90)),
                "P75": float(np.percentile(arr_np, 75)),
                "P50": float(np.percentile(arr_np, 50)),
                "P25": float(np.percentile(arr_np, 25)),
                "Std": float(arr_np.std(ddof=1) if len(arr_np) > 1 else 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values("P90", ascending=False)


def main() -> Optional[str]:
    args = parse_args()
    cfg = load_config(args.config)
    settings = merge_settings(args, cfg)
    date = resolve_date(args, cfg)

    read_contest_meta(args.contest)

    projections = load_labs_for_date(args.labs_dir, date)
    pool = prepare_player_pool(projections)

    sims = int(cfg.get("sims", args.sims))
    out_path = args.out or cfg.get("out") or f"out/sim_{date}.csv"
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    report = simulate_lineups(args.lineups, pool, sims, settings)
    report.to_csv(out_path, index=False)
    log.info("Wrote %s", out_path)
    return out_path


if __name__ == "__main__":
    main()

