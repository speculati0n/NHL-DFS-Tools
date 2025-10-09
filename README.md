# NHL-DFS-Tools (CLI only)

Pure-CLI NHL optimizer + simulator using FantasyLabs CSVs as the projections source.
No overlap with NFL code. DraftKings Classic roster: C,C,W,W,W,D,D,G,UTIL (cap=50000).

## Quickstart
python -m venv .venv && . .venv/Scripts/activate  # on Windows
pip install -r requirements.txt

# OPTIMIZER (build 50 lineups for a given slate date)
python cli/nhl_opt.py ^
  --date 2021-10-13 ^
  --out "out/lineups_2021-10-13.csv" ^
  --num-lineups 50 ^
  --evline2 1 --pp1_2 1 --bringback 1 ^
  --max-vs-goalie 0 ^
  --min-salary 49500 --diversify 2

# SIMULATOR (Monte Carlo the resulting CSV)
python cli/nhl_sim.py ^
  --lineups "out/lineups_2021-10-13.csv" ^
  --date 2021-10-13 ^
  --sims 20000 ^
  --out "out/sim_report_2021-10-13.csv"

## FantasyLabs inputs
By default the CLIs look for FantasyLabs exports inside the repository's `dk_data` folder.
Populate that directory (or point `--labs-dir` elsewhere) with these four files for the slate date:
  fantasylabs_player_data_NHL_C_YYYYMMDD.csv
  fantasylabs_player_data_NHL_W_YYYYMMDD.csv
  fantasylabs_player_data_NHL_D_YYYYMMDD.csv
  fantasylabs_player_data_NHL_G_YYYYMMDD.csv

We read: Player, Team, Opp, Salary, Proj (and optionally Own), Full, PP.
- `Full` like "1F", "2F", "3F" (forwards) or "1D", "2D" (defense)
- `PP` 1 or 2 for power-play units

## Stacks supported
- SameTeam_2+, SameTeam_3+, SameTeam_4+, SameTeam_5+
- EVLine_2+, EVLine_3+ (e.g., BOS L1 forwards / D-pair)
- PP1_2+, PP1_3+, PP2_2+
- BringBack_1+ (game-level: if you stack a team ≥2, require ≥1 skater from the opponent for at least K games)
- Max skaters vs own G

## Consistency / Upside / Duds
Objective:  Score = Proj * (1 + w_up*UpsideZ) * (1 + w_con*ConsistencyZ) - w_dud*DudPenalty
Tune via CLI flags: --w-up, --w-con, --w-dud, --ceil-mult, --floor-mult.
