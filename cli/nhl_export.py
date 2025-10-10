#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys

def _ensure_repo():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root not in sys.path:
        sys.path.insert(0, root)
_ensure_repo()

from nhl_tools import dk_export

def main():
    ap = argparse.ArgumentParser(description="Export NHL lineups → DK wide CSV (Name (playerid))")
    ap.add_argument("--lineups", required=True, help="Input lineups CSV (tall or wide)")
    ap.add_argument("--player-ids", required=True, help="Path to NHL dk_data/player_ids.csv")
    ap.add_argument("--out", required=True, help="Output DK-format CSV")
    ap.add_argument("--strict", action="store_true", help="Fail if any player lacks an ID")
    args = ap.parse_args()

    dk_export.export(args.lineups, args.player_ids, args.out, strict=args.strict)
    print(f"[NHL][Export] wrote {args.out}")

if __name__ == "__main__":
    main()
