#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for the NHL GPP simulator (NFL-style interface)."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence


def _ensure_repo_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edges-aware NHL GPP simulator")
    parser.add_argument("--site", default="DK", help="Site to simulate (default: DK)")
    parser.add_argument("--field-size", type=int, default=20000, help="Contest field size")
    parser.add_argument("--iterations", type=int, default=5000, help="Simulation iterations")
    parser.add_argument("--lineups", required=True, help="CSV of generated lineups")
    parser.add_argument(
        "--players",
        help="Optional player pool CSV (FantasyLabs export or optimizer pool)",
    )
    parser.add_argument(
        "--ownership-file",
        help="Optional ownership CSV to join on (Name, TeamAbbrev, Position)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--outdir",
        default="output",
        help="Output directory for simulator CSVs (default: output/)",
    )
    parser.add_argument(
        "--config",
        default="config/nhl_edges.yaml",
        help="YAML config with ownership/stack tuning",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging (only errors)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    _ensure_repo_on_path()
    args = _parse_args(argv)

    from nhl_tools.nhl_simulator import main as sim_main

    sim_main(args)


if __name__ == "__main__":
    main()
