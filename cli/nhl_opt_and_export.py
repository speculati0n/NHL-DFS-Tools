#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return repo_root


_repo_root = _ensure_repo_on_path()

# Local imports after repo on path
from nhl_tools import dk_export


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run NHL optimizer then immediately export DK upload file."
    )
    ap.add_argument("--labs-dir", type=str, default=None, help="dk_data directory")
    ap.add_argument("--date", type=str, required=True, help="Slate date YYYY-MM-DD")
    ap.add_argument("--out", type=str, required=True, help="Optimizer (tall) CSV output path")
    ap.add_argument("--num-lineups", type=int, default=None, help="Override num_lineups in config")
    ap.add_argument("--num-uniques", type=int, default=None, help="Override num_uniques in config")

    # Export controls
    ap.add_argument("--player-ids", type=str, default=None,
                    help="Path to NHL dk_data/player_ids.csv (default: <labs-dir>/player_ids.csv)")
    ap.add_argument("--export-out", type=str, required=False,
                    help="Path for DK-upload CSV (wide). Defaults to <out> with _DKUPLOAD suffix.")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any player cannot be mapped")
    return ap.parse_args(argv)


def _default_player_ids(labs_dir: Optional[str]) -> str:
    if labs_dir:
        path = os.path.join(labs_dir, "player_ids.csv")
        if os.path.exists(path):
            return path
    # fallback to repo dk_data
    fallback = _repo_root / "dk_data" / "player_ids.csv"
    return str(fallback)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    # Run optimizer (writes tall CSV at args.out)
    # We forward CLI args into nhl_opt via environment overrides recognized by nhl_opt.py
    # But simpler: call nhl_optimizer.main-like function via nhl_opt CLI by importing:
    # The nhl_tools.nhl_optimizer.main signature expects kwargs provided by nhl_opt.py.
    # Here we mirror the minimal required args by writing a small shim call in-process.

    # Build a minimal config override via environment variables nhl_opt.py consumes.
    # Alternatively, we can directly call into nhl_tools.nhl_optimizer.main via a tiny glue:
    # We will import here to avoid circulars; already imported as opt_main.
    # Call with lightweight kwargs (nhl_opt.py writes YAML normally; nhl_optimizer.main accepts kwargs).
    # For safety, we just shell through nhl_opt CLI programmatically:
    import subprocess, sys as _sys

    cmd = [
        _sys.executable, "-m", "cli.nhl_opt",
        "--date", args.date,
        "--out", args.out,
    ]
    if args.labs_dir:
        cmd += ["--labs-dir", args.labs_dir]
    if args.num_lineups is not None:
        cmd += ["--num-lineups", str(args.num_lineups)]
    if args.num_uniques is not None:
        cmd += ["--min-uniques", str(args.num_uniques)]

    print("[NHL] Running optimizer:", " ".join(cmd))
    ret = subprocess.run(cmd, check=False)
    if ret.returncode != 0:
        raise SystemExit(f"Optimizer failed (exit {ret.returncode}).")

    # Now export to DK format (wide, Name (id) cells)
    player_ids = args.player_ids or _default_player_ids(args.labs_dir)
    export_out = args.export_out or os.path.splitext(args.out)[0] + "_DKUPLOAD.csv"

    # Use the robust exporter that accepts tall or wide input
    print(f"[NHL] Exporting DK upload: {export_out}")
    dk_export.export(
        lineups_path=args.out,
        ids_path=player_ids,
        out_path=export_out,
        strict=args.strict,
        league="NHL",
    )
    print("[NHL] Done.")


if __name__ == "__main__":
    main()
