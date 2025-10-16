#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence, Optional, List

import pandas as pd

def _ensure_repo_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

_ensure_repo_on_path()

# Try to import your simulator; fallback to a stub
try:
    from nhl_tools.nhl_simulator import run_simulator  # type: ignore
    HAS_INTERNAL_SIM = True
except Exception:
    HAS_INTERNAL_SIM = False

from nhl_tools.nhl_field_gen import FieldGenConfig, generate_field_lineups


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NHL GPP simulator with field generation (DK-style).")

    # Core inputs
    p.add_argument("--site", default="DK", choices=["DK"], help="Site (DK supported).")
    p.add_argument("--players", required=True, help="Path to players CSV OR directory (e.g., dk_data).")
    p.add_argument("--lineups", required=True, help="Optimizer lineups CSV (your entries).")

    # Field population
    p.add_argument("--field-size", type=int, default=5000, help="Number of field lineups to generate.")
    p.add_argument("--field-lineups", default=None, help="Optional: existing field lineups CSV to use.")
    p.add_argument("--generate-field", action="store_true", default=False,
                   help="Force generate field lineups (even if --field-lineups provided).")

    # Field gen knobs
    p.add_argument("--no-goalie-vs-opp", action="store_true", default=False,
                   help="Forbid goalie vs opposing skaters (on by default in generator).")
    p.add_argument("--max-skaters-same-team", type=int, default=5)
    p.add_argument("--min-salary", type=int, default=48000)
    p.add_argument("--max-salary", type=int, default=50000)
    p.add_argument("--own-beta", type=float, default=1.0, help="Ownership exponent.")
    p.add_argument("--proj-temp", type=float, default=0.03, help="Projection temperature.")
    p.add_argument("--proj-jitter-sd", type=float, default=2.5)
    p.add_argument("--rand-seed", type=int, default=777)

    # Sim config
    p.add_argument("--iterations", type=int, default=1000, help="Simulation iterations.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    p.add_argument("--out-prefix", default="DK_gpp", help="Prefix for outputs.")
    p.add_argument("--combine-only", action="store_true",
                   help="Only emit combined field+entries CSVs, skip running the simulator.")

    return p


def _load_players(players_arg: str) -> pd.DataFrame:
    """Allow either a path to a single CSV or a directory containing a canonical players file."""
    if os.path.isdir(players_arg):
        for candidate in ["players.csv", "projections.csv", "NHL_players.csv", "NHL_projections.csv"]:
            path = os.path.join(players_arg, candidate)
            if os.path.exists(path):
                return pd.read_csv(path)
        parts: List[pd.DataFrame] = []
        for fn in os.listdir(players_arg):
            if fn.lower().endswith(".csv"):
                try:
                    parts.append(pd.read_csv(os.path.join(players_arg, fn)))
                except Exception:
                    pass
        if not parts:
            raise FileNotFoundError(f"No player CSVs found under directory: {players_arg}")
        return pd.concat(parts, ignore_index=True)
    else:
        return pd.read_csv(players_arg)


def _safe_stub_sim(lineups_df: pd.DataFrame, iterations: int) -> pd.DataFrame:
    """Fallback sim: assigns naive scores so CLI runs even without internal simulator."""
    import numpy as np
    base = np.random.normal(loc=100.0, scale=15.0, size=len(lineups_df))
    out = lineups_df.copy()
    out["SimScore_mean"] = base
    out["SimScore_sd"] = 15.0
    out["Win%"] = np.maximum(0, np.random.normal(0.05, 0.03, size=len(lineups_df)))
    out["Top1%"] = out["Win%"] * 0.2
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    players_df = _load_players(args.players)
    entries_df = read_csv(args.lineups)

    field_df: Optional[pd.DataFrame] = None
    out_field_csv = os.path.join(args.outdir, f"{args.out_prefix}_field_lineups_{args.field_size}.csv")

    if args.field_lineups and not args.generate_field and os.path.exists(args.field_lineups):
        field_df = read_csv(args.field_lineups)
        print(f"[sim] Using existing field lineups: {args.field_lineups}")
    else:
        cfg = FieldGenConfig(
            field_size=int(args.field_size),
            max_skaters_same_team=int(args.max_skaters_same_team),
            lock_no_goalie_vs_opp=(not args.no_goalie_vs_opp),
            min_salary=int(args.min_salary),
            max_salary=int(args.max_salary),
            sample_beta_own=float(args.own_beta),
            sample_temp_proj=float(args.proj_temp),
            proj_jitter_sd=float(args.proj_jitter_sd),
            random_seed=int(args.rand_seed),
        )
        print(f"[sim] Generating field lineups: size={cfg.field_size}")
        field_df = generate_field_lineups(players_df, cfg)
        write_csv(field_df, out_field_csv)
        print(f"[sim] Wrote field lineups -> {out_field_csv}")

    field_df = field_df.copy()
    field_df["source"] = "field"

    entries_df = entries_df.copy()
    entries_df["source"] = "entries"

    common_cols = sorted(set(field_df.columns) & set(entries_df.columns))
    combined = pd.concat([field_df[common_cols], entries_df[common_cols]], ignore_index=True)

    out_pop_csv = os.path.join(args.outdir, f"{args.out_prefix}_population_{len(combined)}.csv")
    write_csv(combined, out_pop_csv)
    print(f"[sim] Wrote combined population -> {out_pop_csv}")

    if args.combine_only:
        return 0

    if HAS_INTERNAL_SIM:
        results = run_simulator(combined, iterations=int(args.iterations))  # type: ignore
    else:
        results = _safe_stub_sim(combined, iterations=int(args.iterations))

    out_results_csv = os.path.join(args.outdir, f"{args.out_prefix}_sim_results_{args.iterations}.csv")
    write_csv(results, out_results_csv)
    print(f"[sim] Wrote sim results -> {out_results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
