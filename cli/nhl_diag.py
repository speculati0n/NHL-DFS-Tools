#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick diagnostics for NHL edges configuration feasibility."""
from __future__ import annotations

import argparse
import logging
from typing import Tuple

import numpy as np

from nhl_tools.nhl_data import DK_SALARY_CAP, group_evline, group_pp1, load_player_pool
from nhl_tools.nhl_optimizer import _load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("nhl_diag")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("NHL diagnostics")
    ap.add_argument("--date", required=True)
    ap.add_argument("--labs-dir", default="dk_data")
    ap.add_argument("--config", default="config/nhl_edges.yaml")
    ap.add_argument("--min-spend", type=int, default=49500)
    return ap.parse_args()


def _approx_salary_bounds(pool) -> Tuple[int, int, bool]:
    by_pos = {pos: pool[pool["Pos"] == pos].sort_values("Salary") for pos in ["C", "W", "D", "G"]}
    min_sal = 0
    max_sal = 0
    feasible = True
    for pos, need in [("C", 2), ("W", 3), ("D", 2), ("G", 1)]:
        df = by_pos[pos]
        if len(df) < need:
            feasible = False
            continue
        min_sal += int(df.head(need)["Salary"].sum())
        max_sal += int(df.tail(need)["Salary"].sum())
    skaters = pool[pool["Pos"].isin(["C", "W", "D"])].sort_values("Salary")
    if len(skaters) >= 1:
        min_sal += int(skaters.iloc[0]["Salary"])
        max_sal += int(skaters.iloc[-1]["Salary"])
    else:
        feasible = False
    return min_sal, max_sal if feasible else 0, feasible


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)
    pool = load_player_pool(args.labs_dir, args.date, cfg.get("ownership"))
    if pool.empty:
        raise SystemExit("Empty player pool.")

    counts = pool.groupby("Pos")["Name"].count().to_dict()
    LOG.info("Player counts: %s", counts)

    ev_groups = group_evline(pool)
    if ev_groups:
        sizes = [len(v) for v in ev_groups.values()]
        LOG.info(
            "EV line groups: total=%d | size dist=%s",
            len(ev_groups),
            dict(zip(*np.unique(sizes, return_counts=True))),
        )
    else:
        LOG.info("No EV groups detected (missing Full column)")

    pp1_groups = group_pp1(pool)
    if pp1_groups:
        sizes = [len(v) for v in pp1_groups.values()]
        LOG.info(
            "PP1 groups: total=%d | size dist=%s",
            len(pp1_groups),
            dict(zip(*np.unique(sizes, return_counts=True))),
        )
    else:
        LOG.info("No PP1 groups detected")

    min_sal, max_sal, feasible = _approx_salary_bounds(pool)
    LOG.info("Approx salary bounds: min=%s, max=%s", min_sal, max_sal)
    min_spend = int(cfg.get("salary", {}).get("min_spend", args.min_spend))
    leftover = int(cfg.get("salary", {}).get("max_leftover", DK_SALARY_CAP - min_sal))
    LOG.info("Config min spend=%d, max leftover=%d", min_spend, leftover)
    if not feasible:
        LOG.warning("Pool missing required positional depth; salary bounds may be inaccurate")
    if min_sal > DK_SALARY_CAP - leftover:
        LOG.warning("Requested min spend may be infeasible with current pool")


if __name__ == "__main__":
    main()
