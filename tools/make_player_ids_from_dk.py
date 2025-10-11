#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a DraftKings export-style CSV into the optimizer's required player ID mapping.

Input (your current DK export):
  columns: Position, Name + ID, Name, ID, Roster Position, Salary, Game Info, TeamAbbrev, AvgPointsPerGame

Output (dk_data/player_ids.csv):
  columns: player_id,name,team,pos

Usage:
  python tools/make_player_ids_from_dk.py --in /path/to/DK_export.csv --out dk_data/player_ids.csv
"""
import argparse, os, sys, csv

def norm_pos(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip().upper()
    for p in ["C","W","D","G"]:
        if p in raw.split("/"):
            return p
    for p in ["C","W","D","G"]:
        if p in raw:
            return p
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Path to DK export (your player_ids.csv)")
    ap.add_argument("--out", dest="out_path", default="dk_data/player_ids.csv", help="Output CSV path")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(in_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = [c.strip() for c in r.fieldnames or []]
        required_src = {"Name","ID","TeamAbbrev"}
        if not required_src.issubset(set(cols)):
            print(f"[ERROR] Missing required columns: found {cols}, need {sorted(required_src)}", file=sys.stderr)
            sys.exit(2)

        rows = []
        for row in r:
            name = (row.get("Name") or "").strip()
            pid  = (row.get("ID") or "").strip()
            team = (row.get("TeamAbbrev") or "").strip().upper()
            raw_pos = (row.get("Position") or row.get("Roster Position") or "").strip()
            pos = norm_pos(raw_pos)
            if not (name and pid and team and pos):
                continue
            rows.append({
                "player_id": pid,
                "name": name,
                "team": team,
                "pos": pos,
            })

    # Deduplicate
    seen = set()
    final = []
    for r in rows:
        key = (r["player_id"], r["name"].lower().strip(), r["team"], r["pos"])
        if key in seen:
            continue
        seen.add(key)
        final.append(r)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["player_id","name","team","pos"])
        w.writeheader()
        for r in final:
            w.writerow(r)

    print(f"[OK] Wrote {len(final)} player mappings → {out_path}")

if __name__ == "__main__":
    main()
