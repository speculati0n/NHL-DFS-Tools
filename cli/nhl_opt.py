import argparse
import os
import sys
from typing import Any, Dict

import yaml


def _ensure_repo_on_path() -> None:
    """Add the repository root to sys.path for direct CLI execution."""

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _default_config_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    primary = os.path.join(repo_root, "configs", "optimizer.yaml")
    fallback = os.path.join(repo_root, "configs", "optimizer.example.yaml")
    return primary if os.path.exists(primary) else fallback


def _set_nested(cfg: Dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


_ensure_repo_on_path()

from nhl_tools.nhl_optimizer import main as optimizer_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NHL Optimizer (DK) — FantasyLabs input.")
    parser.add_argument("--labs-dir", default=None, help="Folder with FantasyLabs NHL CSVs")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD slate date")
    parser.add_argument("--out", required=True, help="Output CSV for lineups")
    parser.add_argument("--num-lineups", type=int, default=20)
    parser.add_argument("--min-salary", type=int, default=49500)

    # Legacy correlation / roster flags
    parser.add_argument("--max-vs-goalie", type=int, default=None,
                        help="Override max skaters vs own goalie constraint")
    parser.add_argument("--evline2", type=int, default=0)
    parser.add_argument("--evline3", type=int, default=0)
    parser.add_argument("--pp1_2", type=int, default=0)
    parser.add_argument("--pp1_3", type=int, default=0)
    parser.add_argument("--pp2_2", type=int, default=0)
    parser.add_argument("--same2", type=int, default=0)
    parser.add_argument("--same3", type=int, default=0)
    parser.add_argument("--same4", type=int, default=0)
    parser.add_argument("--same5", type=int, default=0)
    parser.add_argument("--bringback", type=int, default=0)

    # Projection quality weights
    parser.add_argument("--w-up", type=float, default=0.15)
    parser.add_argument("--w-con", type=float, default=0.05)
    parser.add_argument("--w-dud", type=float, default=0.03)
    parser.add_argument("--ceil-mult", type=float, default=1.6)
    parser.add_argument("--floor-mult", type=float, default=0.55)

    parser.add_argument("--diversify", type=int, default=1,
                        help=">0 to force at least one change from prior lineup (legacy)")

    # Config management & overrides
    parser.add_argument("--config", default=None, help="Path to optimizer YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--leftover-mode", choices=["none", "light", "mix", "aggressive"], default=None)
    parser.add_argument("--total-own-max", type=float, default=None)
    parser.add_argument("--chalk-thresh", type=float, default=None)
    parser.add_argument("--max-chalk", type=int, default=None)
    parser.add_argument("--w-eff", type=float, default=None)
    parser.add_argument("--w-ev", type=float, default=None,
                        help="Override EV stack (3+) objective weight")
    parser.add_argument("--w-pp1", type=float, default=None,
                        help="Override PP1 stack (3+) objective weight")
    parser.add_argument("--min-uniques", type=int, default=None)
    parser.add_argument("--bringbacks-max", type=int, default=None)
    parser.add_argument("--objective-noise-std", type=float, default=None)
    parser.add_argument("--weight-jitter-pct", type=float, default=None)

    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg_path = args.config or _default_config_path()
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    overrides: list[tuple[tuple[str, ...], Any]] = []
    if args.seed is not None:
        overrides.append((("random", "seed"), args.seed))
    if args.leftover_mode:
        overrides.append((("random", "leftover_mode"), args.leftover_mode))
    if args.total_own_max is not None:
        overrides.append((("ownership", "total_own_max"), args.total_own_max))
    if args.chalk_thresh is not None:
        overrides.append((("ownership", "chalk_thresh"), args.chalk_thresh))
    if args.max_chalk is not None:
        overrides.append((("ownership", "max_chalk"), args.max_chalk))
    if args.w_eff is not None:
        overrides.append((("ownership", "w_eff"), args.w_eff))
    if args.w_ev is not None:
        overrides.append((("correlation", "w_ev_stack_3p"), args.w_ev))
    if args.w_pp1 is not None:
        overrides.append((("correlation", "w_pp1_stack_3p"), args.w_pp1))
    if args.min_uniques is not None:
        overrides.append((("uniqueness", "min_uniques"), args.min_uniques))
    if args.bringbacks_max is not None:
        overrides.append((("correlation", "bringbacks_max"), args.bringbacks_max))
    if args.max_vs_goalie is not None:
        overrides.append((("correlation", "max_skaters_vs_goalie"), args.max_vs_goalie))
    if args.objective_noise_std is not None:
        overrides.append((("random", "objective_noise_std"), args.objective_noise_std))
    if args.weight_jitter_pct is not None:
        overrides.append((("random", "weight_jitter_pct"), args.weight_jitter_pct))

    for path, value in overrides:
        _set_nested(cfg, path, value)

    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args)

    labs_dir = args.labs_dir or os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "dk_data")
    max_vs_goalie_cfg = cfg.get("correlation", {}).get("max_skaters_vs_goalie")
    if args.max_vs_goalie is not None:
        max_vs_goalie = args.max_vs_goalie
    elif max_vs_goalie_cfg is not None:
        max_vs_goalie = max_vs_goalie_cfg
    else:
        max_vs_goalie = 0

    optimizer_main(
        cfg=cfg,
        labs_dir=labs_dir,
        date=args.date,
        out=args.out,
        num_lineups=args.num_lineups,
        min_salary=args.min_salary,
        max_vs_goalie=max_vs_goalie,
        evline2=args.evline2,
        evline3=args.evline3,
        pp1_2=args.pp1_2,
        pp1_3=args.pp1_3,
        pp2_2=args.pp2_2,
        same2=args.same2,
        same3=args.same3,
        same4=args.same4,
        same5=args.same5,
        bringback=args.bringback,
        w_up=args.w_up,
        w_con=args.w_con,
        w_dud=args.w_dud,
        ceil_mult=args.ceil_mult,
        floor_mult=args.floor_mult,
        diversify=args.diversify,
    )


if __name__ == "__main__":
    main()
