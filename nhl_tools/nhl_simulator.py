#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations


import argparse
import json
import logging
import math
import os
import re


import numpy as np
import pandas as pd
from scipy.stats import lognorm


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



        lid = int(row["LineupID"])
        for slot in slot_cols:
            val = row.get(slot)
            if pd.isna(val):
                continue
            name_str = str(val).strip()
            if not name_str:
                continue

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


    for _ in range(int(sims)):
        sampled = enriched.copy()
        sampled["Pts"] = sampled.apply(lambda r: _sample_points(r["Mean"], r["Vol"], rng), axis=1)
        totals = sampled.groupby("LineupID")["Pts"].sum()
        for lid, val in totals.items():
            results[lid].append(float(val))

    rows = []
    for lid, arr in results.items():

            }
        )
    return pd.DataFrame(rows).sort_values("P90", ascending=False)




if __name__ == "__main__":
    main()

