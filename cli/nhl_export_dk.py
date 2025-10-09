"""CLI for exporting NHL lineups to DraftKings upload format."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional


def _ensure_repo_on_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NHL lineups to DraftKings upload format.")
    parser.add_argument("--lineups", required=True, help="Path to optimizer output lineups CSV")
    parser.add_argument(
        "--player-ids",
        default="dk_data/player_ids.csv",
        help="CSV with DraftKings player IDs (default: dk_data/player_ids.csv)",
    )
    parser.add_argument("--out", required=True, help="Output CSV path for DraftKings upload")
    parser.add_argument("--league", default="NHL", help="League guard (only NHL supported)")
    parser.add_argument("--strict", action="store_true", help="Fail if any player cannot be mapped")
    parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Print the first two formatted lineups after export",
    )
    return parser.parse_args(argv)


_repo_root = _ensure_repo_on_path()

from nhl_tools import dk_export


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.league.upper() != "NHL":
        raise SystemExit("Only NHL league is supported by this exporter.")

    try:
        out_df = dk_export.export(
            lineups_path=args.lineups,
            ids_path=args.player_ids,
            out_path=args.out,
            strict=args.strict,
            league=args.league,
        )
    except (RuntimeError, ValueError) as err:
        logging.error("%s", err)
        raise SystemExit(1) from err

    if args.show_sample:
        sample = out_df.head(2)
        if not sample.empty:
            csv_text = sample.to_csv(index=False)
            print(csv_text.strip())
        else:
            print("No lineups exported.")


if __name__ == "__main__":
    main()
