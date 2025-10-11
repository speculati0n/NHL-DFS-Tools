#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edges-aware lineup simulator and diagnostics."""
from __future__ import annotations

import argparse
import logging
import math
import os
import unicodedata
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from nhl_tools.nhl_data import DK_SALARY_CAP, load_player_pool
from nhl_tools.nhl_optimizer import _load_config

LOG = logging.getLogger("nhl_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SLOT_ORDER = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Edges-aware NHL lineup simulator")
    ap.add_argument("--lineups", required=True, help="CSV produced by the optimizer")
    ap.add_argument("--date", required=True)
    ap.add_argument("--labs-dir", default="dk_data")
    ap.add_argument("--config", default="config/nhl_edges.yaml")
    ap.add_argument("--results", default="out/sim_results_{date}.csv")
    return ap.parse_args()


def _load_lineups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    available = [c for c in SLOT_ORDER if c in df.columns]
    if len(available) != len(SLOT_ORDER):
        missing = [c for c in SLOT_ORDER if c not in available]
        raise ValueError(f"Lineup file missing slots: {missing}")
    return df


def _match_players(pool: pd.DataFrame, names: Sequence[str], slots: Sequence[str]) -> List[pd.Series]:
    lookup = pool.copy()
    lookup["NameNorm"] = lookup["Name"].map(_normalize_name)
    matched: List[pd.Series] = []
    for slot, name in zip(slots, names):
        nm = str(name or "").strip()
        if not nm:
            continue
        nm_norm = _normalize_name(nm)
        cand = lookup[lookup["NameNorm"] == nm_norm]
        if cand.empty:
            LOG.warning("Unable to match %s for slot %s", nm, slot)
            continue
        picked = cand.iloc[0]
        matched.append(picked)
    return matched


def _primary_team(players: List[pd.Series]) -> Tuple[str, float]:
    team_counts: Dict[str, List[pd.Series]] = {}
    for row in players:
        if not row.get("IsSkater", False):
            continue
        team = row.get("Team")
        team_counts.setdefault(team, []).append(row)
    if not team_counts:
        return "", 0.0
    # Choose largest stack, tie-break by projection sum
    best_team = None
    best_size = -1
    best_proj = -1.0
    for team, rows in team_counts.items():
        size = len(rows)
        proj = sum(float(r.get("Proj", 0.0)) for r in rows)
        if size > best_size or (size == best_size and proj > best_proj):
            best_team = team
            best_size = size
            best_proj = proj
    return best_team or "", best_proj


def _cluster_size(players: List[pd.Series], key_fn) -> int:
    buckets: Dict[str, int] = {}
    for row in players:
        if not row.get("IsSkater", False):
            continue
        key = key_fn(row)
        if not key:
            continue
        buckets[key] = buckets.get(key, 0) + 1
    return max(buckets.values()) if buckets else 0


def _lineup_metrics(players: List[pd.Series]) -> Dict[str, object]:
    salary = sum(int(row.get("Salary", 0)) for row in players)
    proj = sum(float(row.get("Proj", 0.0)) for row in players)
    ownership = [float(row.get("Ownership", np.nan)) for row in players if not math.isnan(float(row.get("Ownership", np.nan)))]
    own_sum = float(np.nansum(ownership)) if ownership else float("nan")
    own_hhi = float(np.nansum((np.array(ownership) / 100.0) ** 2)) if ownership else float("nan")

    primary_team, _ = _primary_team(players)
    primary_opponents = {
        str(row.get("Opp"))
        for row in players
        if row.get("Team") == primary_team and isinstance(row.get("Opp"), str)
    }
    opponent = next(iter(primary_opponents)) if primary_opponents else ""

    bringback_count = sum(1 for row in players if row.get("Team") == opponent and row.get("IsSkater", False))

    goalie_team = next((row.get("Team") for row in players if row.get("Pos") == "G"), "")
    goalie_conflict = sum(1 for row in players if row.get("Opp") == goalie_team and row.get("IsSkater", False))

    ev_cluster = _cluster_size(players, lambda r: f"{r.get('Team')}:{r.get('Full')}" if r.get("Full") else "")
    pp1_cluster = _cluster_size(players, lambda r: r.get("Team") if r.get("pp_unit") == 1 else "")

    signature = "|".join(sorted(str(row.get("Name")) for row in players))
    leftover = DK_SALARY_CAP - salary

    return {
        "salary": salary,
        "projection": proj,
        "ownership_sum": own_sum,
        "ownership_hhi": own_hhi,
        "ev_cluster": ev_cluster,
        "pp1_cluster": pp1_cluster,
        "bringback_count": bringback_count,
        "goalie_conflict": goalie_conflict,
        "salary_leftover": leftover,
        "primary_team": primary_team,
        "primary_opponent": opponent,
        "signature": signature,
    }


def _adjusted_score(metrics: Dict[str, object], cfg: Dict) -> float:
    score = float(metrics.get("projection", 0.0))
    ev_bonus = max(0.0, float(metrics.get("ev_cluster", 0)) - 2.0)
    pp1_bonus = max(0.0, float(metrics.get("pp1_cluster", 0)) - 2.0)
    goalie_pen = float(metrics.get("goalie_conflict", 0))
    bringback_pen = float(metrics.get("bringback_count", 0))
    leftover_pen = max(0.0, float(metrics.get("salary_leftover", 0)) - 500.0)

    score += 0.5 * ev_bonus
    score += 0.4 * pp1_bonus
    score -= 3.0 * goalie_pen
    score -= 1.0 * min(2.0, bringback_pen)
    score -= 0.1 * (leftover_pen / 100.0)

    own_cfg = cfg.get("ownership", {})
    if own_cfg.get("enable"):
        max_sum = own_cfg.get("max_sum")
        max_hhi = own_cfg.get("max_hhi")
        own_sum = metrics.get("ownership_sum")
        own_hhi = metrics.get("ownership_hhi")
        if isinstance(own_sum, float) and not math.isnan(own_sum) and max_sum is not None and own_sum > max_sum:
            score -= 1.0
        if isinstance(own_hhi, float) and not math.isnan(own_hhi) and max_hhi is not None and own_hhi > max_hhi:
            score -= 0.5
    return score


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    out_path = args.results.format(date=args.date)

    pool = load_player_pool(args.labs_dir, args.date, cfg.get("ownership"))
    if pool.empty:
        raise SystemExit("Empty player pool")

    pool = pool.copy()
    pool["NameNorm"] = pool["Name"].map(_normalize_name)

    lineups_df = _load_lineups(args.lineups)
    results = []

    for _, row in lineups_df.iterrows():
        names = [row.get(slot) for slot in SLOT_ORDER]
        matched = _match_players(pool, names, SLOT_ORDER)
        if len(matched) != len(SLOT_ORDER):
            LOG.warning("Lineup missing matches for some slots; skipping")
            continue
        metrics = _lineup_metrics(matched)
        metrics["sim_score"] = _adjusted_score(metrics, cfg)
        metrics["LineupID"] = int(row.get("LineupID", len(results) + 1))
        results.append(metrics)

    if not results:
        raise SystemExit("No lineups could be evaluated")

    out_df = pd.DataFrame(results)
    out_df = out_df[
        [
            "LineupID",
            "sim_score",
            "projection",
            "salary",
            "salary_leftover",
            "ev_cluster",
            "pp1_cluster",
            "bringback_count",
            "goalie_conflict",
            "ownership_sum",
            "ownership_hhi",
            "primary_team",
            "primary_opponent",
            "signature",
        ]
    ]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
