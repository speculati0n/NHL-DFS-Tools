#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, logging
from nhl_tools.nhl_optimizer import load_labs_merged, _approx_feasible_salary_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("nhl_diag")

def parse_args():
    ap = argparse.ArgumentParser("NHL diag")
    ap.add_argument("--date", required=True)
    ap.add_argument("--labs-dir", default="dk_data")
    ap.add_argument("--min-salary", type=int, default=48000)
    ap.add_argument("--max-salary", type=int, default=50000)
    return ap.parse_args()

def main():
    a = parse_args()
    df = load_labs_merged(a.labs_dir, a.date)
    counts = df.groupby("pos")["name"].count().to_dict()
    LOG.info("Counts per pos: %s", counts)
    feas_min, feas_max, ok, _ = _approx_feasible_salary_range(df)
    LOG.info("Feasible salary range (approx): min=%d, max=%d", feas_min, feas_max)
    LOG.info("Requested salary window: [%d, %d]", a.min_salary, a.max_salary)
    for p in ["G","C","D","W"]:
        sub = df[df["pos"]==p]["salary"]
        if len(sub):
            LOG.info("%s salary min=%s, max=%s, median=%s (n=%d)", p, sub.min(), sub.max(), int(sub.median()), len(sub))
    if not ok:
        LOG.info("Pool incomplete — check missing positions.")
    else:
        LOG.info("Pool looks feasible.")
if __name__ == "__main__":
    main()
