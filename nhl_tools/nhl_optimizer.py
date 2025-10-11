#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Edges-aware NHL optimizer with stack templates and ownership controls."""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pulp

from nhl_tools.nhl_data import (
    DK_ROSTER,
    DK_SALARY_CAP,
    group_evline,
    group_pp1,
    load_player_pool,
)
from nhl_tools.nhl_export import export_dk_upload

LOG = logging.getLogger("nhl_opt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ALPHA_STACK_BONUS = 0.25
BETA_GOALIE_CONFLICT = 100.0
GAMMA_BRINGBACK = 25.0
DELTA_OWNERSHIP = 0.1


@dataclass
class ShapePlan:
    counts: List[int]
    primary_mode: str = "ev"  # "ev" or "pp1"


@dataclass
class LineupResult:
    indices: List[int]
    slots: Dict[str, int]
    salary: int
    projection: float
    projection_base: float
    ownership_sum: float
    ownership_hhi: float
    stack_details: Dict[str, object] = field(default_factory=dict)

    def signature(self, names: Sequence[str]) -> str:
        picked = [names[i] for i in self.indices]
        return "|".join(sorted(picked))


def _load_config(path: Optional[str]) -> Dict:
    if not path:
        default_path = os.path.join("config", "nhl_edges.yaml")
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as fh:
                try:
                    import yaml  # type: ignore

                    return yaml.safe_load(fh) or {}
                except ImportError as exc:
                    raise RuntimeError("PyYAML is required to read YAML configs") from exc
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        if path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore

                return yaml.safe_load(fh) or {}
            except ImportError as exc:
                raise RuntimeError("PyYAML is required to read YAML configs") from exc
        import json

        return json.load(fh)


def _parse_shape_arg(shape_str: Optional[str]) -> Optional[List[int]]:
    if not shape_str:
        return None
    parts = []
    for token in shape_str.split("-"):
        token = token.strip()
        if not token:
            continue
        try:
            parts.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid shape token '{token}'. Use e.g. 4-3-1")
    if sum(parts) != 8:
        raise ValueError("Team shape must sum to 8 skaters (e.g. 4-3-1)")
    return parts


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Name", "Pos", "Team", "Opp", "Salary", "Proj"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Player pool missing required column {col}")
    df = df.copy().reset_index(drop=True)
    df["Proj"] = pd.to_numeric(df["Proj"], errors="coerce").fillna(0.0)
    df["Ceiling"] = pd.to_numeric(df.get("Ceiling"), errors="coerce")
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce").fillna(0).astype(int)
    df["Ownership"] = pd.to_numeric(df.get("Ownership"), errors="coerce")
    df["IsSkater"] = df.get("IsSkater", False).astype(bool)
    df["IsGoalie"] = df.get("IsGoalie", False).astype(bool)
    df["pp_unit"] = pd.to_numeric(df.get("pp_unit"), errors="coerce")
    return df


def _ownership_fraction(val: float) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return max(0.0, float(val)) / 100.0


class EdgesOptimizer:
    def __init__(
        self,
        pool: pd.DataFrame,
        config: Dict,
        shapes: Sequence[ShapePlan],
        *,
        tie_break_random_pct: float,
        bringbacks_enabled: bool,
        max_from_opponent: int,
        forbid_goalie_conflict: bool,
        salary_min: int,
        salary_max: int,
        ownership_limits: Dict[str, float],
    ) -> None:
        self.pool = _ensure_columns(pool)
        self.config = config
        self.shapes = list(shapes)
        self.tie_break_random_pct = tie_break_random_pct
        self.bringbacks_enabled = bringbacks_enabled
        self.max_from_opponent = max_from_opponent
        self.forbid_goalie_conflict = forbid_goalie_conflict
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.ownership_limits = ownership_limits

        self.pool["PlayerIndex"] = self.pool.index
        self.pool["ProjBase"] = self.pool["Proj"].astype(float)

        self.team_players: Dict[str, List[int]] = {}
        self.team_skaters: Dict[str, List[int]] = {}
        for idx, row in self.pool.iterrows():
            team = row["Team"]
            self.team_players.setdefault(team, []).append(idx)
            if row["IsSkater"]:
                self.team_skaters.setdefault(team, []).append(idx)

        self.ev_groups = group_evline(self.pool)
        self.pp1_groups = group_pp1(self.pool)

        self.goalies = [idx for idx, row in self.pool.iterrows() if row["IsGoalie"]]
        self.skaters = [idx for idx, row in self.pool.iterrows() if row["IsSkater"]]

        self.opponents_by_team: Dict[str, List[int]] = {}
        for idx, row in self.pool.iterrows():
            opp = row.get("Opp")
            if isinstance(opp, str) and opp:
                self.opponents_by_team.setdefault(opp, []).append(idx)

        self.best_projection: Optional[float] = None

    # ------------------------------------------------------------------
    # ILP construction helpers
    # ------------------------------------------------------------------
    def _build_problem(
        self,
        plan: ShapePlan,
        projections: Dict[int, float],
        additional_cuts: List[Tuple[List[int], int]],
    ) -> Optional[pulp.LpProblem]:
        players = list(self.pool.index)
        prob = pulp.LpProblem("nhl_edges", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in players}

        # Salary window
        prob += pulp.lpSum(self.pool.loc[i, "Salary"] * x[i] for i in players) >= self.salary_min
        prob += pulp.lpSum(self.pool.loc[i, "Salary"] * x[i] for i in players) <= self.salary_max

        # Roster structure
        prob += pulp.lpSum(x[i] for i in players) == 9
        prob += pulp.lpSum(x[i] for i in self.goalies) == 1
        prob += pulp.lpSum(x[i] for i in self.skaters) == 8

        def pos_filter(pos: str) -> Iterable[int]:
            return [i for i in players if self.pool.loc[i, "Pos"] == pos]

        prob += pulp.lpSum(x[i] for i in pos_filter("C")) >= DK_ROSTER["C"]
        prob += pulp.lpSum(x[i] for i in pos_filter("C")) <= DK_ROSTER["C"] + 1
        prob += pulp.lpSum(x[i] for i in pos_filter("W")) >= DK_ROSTER["W"]
        prob += pulp.lpSum(x[i] for i in pos_filter("W")) <= DK_ROSTER["W"] + 1
        prob += pulp.lpSum(x[i] for i in pos_filter("D")) >= DK_ROSTER["D"]
        prob += pulp.lpSum(x[i] for i in pos_filter("D")) <= DK_ROSTER["D"] + 1

        # Goalie vs opponents
        if self.forbid_goalie_conflict:
            for gid in self.goalies:
                team = self.pool.loc[gid, "Team"]
                opp_players = [i for i in self.skaters if self.pool.loc[i, "Opp"] == team]
                for sid in opp_players:
                    prob += x[gid] + x[sid] <= 1

        # Team shape assignments
        slot_sizes = [c for c in plan.counts if c > 1]
        teams = sorted(self.team_skaters.keys())
        if not slot_sizes:
            LOG.error("Shape %s has no stack components.", plan.counts)
            return None

        assign: Dict[Tuple[str, int], pulp.LpVariable] = {}
        for j, size in enumerate(slot_sizes):
            for team in teams:
                assign[(team, j)] = pulp.LpVariable(f"team_slot_{j}_{team}", lowBound=0, upBound=1, cat="Binary")
            prob += pulp.lpSum(assign[(team, j)] for team in teams) == 1
        for team in teams:
            prob += pulp.lpSum(assign[(team, j)] for j in range(len(slot_sizes))) <= 1

        team_skaters_expr = {
            team: pulp.lpSum(x[i] for i in self.team_skaters.get(team, []))
            for team in teams
        }

        big_m = 8
        for team in teams:
            for j, size in enumerate(slot_sizes):
                expr = team_skaters_expr[team]
                prob += expr >= size * assign[(team, j)]
                prob += expr <= size + big_m * (1 - assign[(team, j)])

        # Bring-back rules
        if not self.bringbacks_enabled:
            for team in teams:
                opps = self.opponents_by_team.get(team)
                if not opps:
                    continue
                expr = pulp.lpSum(x[i] for i in opps if self.pool.loc[i, "IsSkater"])
                limit = self.max_from_opponent
                prob += expr <= limit * assign[(team, 0)] + big_m * (1 - assign[(team, 0)])

        # Primary EV / PP1 requirements
        stacks_cfg = self.config.get("stacks", {})
        primary_cfg = stacks_cfg.get("primary_evline", {})
        secondary_cfg = stacks_cfg.get("secondary_pp1", {})
        pp1_full_cfg = stacks_cfg.get("pp1_full_unit", {})

        ev_min = int(primary_cfg.get("min_size", 3))
        ev_max = int(primary_cfg.get("max_size", 4))
        pp1_min = int(secondary_cfg.get("min_size", 2))
        pp1_max = int(secondary_cfg.get("max_size", 3))
        pp1_full_min = int(pp1_full_cfg.get("min_size", 4))
        pp1_full_max = int(pp1_full_cfg.get("max_size", 5))

        ev_vars: Dict[Tuple[str, str], pulp.LpVariable] = {}
        for (team, full), idxs in self.ev_groups.items():
            if len(idxs) < ev_min:
                continue
            var = pulp.LpVariable(f"ev_{team}_{full}", lowBound=0, upBound=1, cat="Binary")
            ev_vars[(team, full)] = var
            expr = pulp.lpSum(x[i] for i in idxs)
            prob += expr >= ev_min * var
            prob += expr <= ev_max + big_m * (1 - var)
            slot_idx = 0 if plan.primary_mode == "ev" else (1 if len(slot_sizes) > 1 else 0)
            if (team, slot_idx) in assign:
                prob += var <= assign[(team, slot_idx)]

        if primary_cfg.get("required", True) and plan.primary_mode == "ev":
            if ev_vars:
                prob += pulp.lpSum(ev_vars.values()) >= 1
            else:
                LOG.warning("No EV stacks satisfy min size %d", ev_min)
                return None

        pp1_vars: Dict[str, pulp.LpVariable] = {}
        for team, idxs in self.pp1_groups.items():
            if len(idxs) < pp1_min:
                continue
            var = pulp.LpVariable(f"pp1_{team}", lowBound=0, upBound=1, cat="Binary")
            pp1_vars[team] = var
            expr = pulp.lpSum(x[i] for i in idxs)
            prob += expr >= pp1_min * var
            prob += expr <= pp1_max + big_m * (1 - var)

        if pp1_vars:
            prob += pulp.lpSum(pp1_vars.values()) >= 1

        pp1_primary_vars: Dict[str, pulp.LpVariable] = {}
        if plan.primary_mode == "pp1":
            for team, idxs in self.pp1_groups.items():
                if len(idxs) < pp1_full_min:
                    continue
                var = pulp.LpVariable(f"pp1_primary_{team}", lowBound=0, upBound=1, cat="Binary")
                pp1_primary_vars[team] = var
                expr = pulp.lpSum(x[i] for i in idxs)
                prob += expr >= pp1_full_min * var
                prob += expr <= pp1_full_max + big_m * (1 - var)
                if (team, 0) in assign:
                    prob += var <= assign[(team, 0)]
            if pp1_primary_vars:
                prob += pulp.lpSum(pp1_primary_vars.values()) >= 1
            else:
                LOG.warning("No PP1 stacks meet full-unit requirement (%d)", pp1_full_min)
                return None
            # Require an EV cluster for secondary slot if available
            if ev_vars:
                prob += pulp.lpSum(ev_vars.values()) >= 1

        # Ownership limits
        own_vals = {i: _ownership_fraction(self.pool.loc[i, "Ownership"]) for i in players}
        own_sum_expr = pulp.lpSum(own_vals[i] * 100.0 * x[i] for i in players)
        own_hhi_expr = pulp.lpSum((own_vals[i] ** 2) * x[i] for i in players)
        max_sum = self.ownership_limits.get("max_sum")
        max_hhi = self.ownership_limits.get("max_hhi")
        if max_sum is not None:
            prob += own_sum_expr <= max_sum
        if max_hhi is not None:
            prob += own_hhi_expr <= max_hhi

        # Objective: base projections + bonuses - penalties
        proj_expr = pulp.lpSum(projections.get(i, 0.0) * x[i] for i in players)
        noise = pulp.lpSum(random.random() * 1e-5 * x[i] for i in players)
        stack_bonus = ALPHA_STACK_BONUS * (
            pulp.lpSum(ev_vars.values()) + pulp.lpSum(pp1_vars.values()) + pulp.lpSum(pp1_primary_vars.values())
        )
        ownership_penalty = DELTA_OWNERSHIP * own_sum_expr
        prob += proj_expr + noise + stack_bonus - ownership_penalty

        # Additional cuts to prevent duplicates / rejected lineups
        for cut_idx, (indices, limit) in enumerate(additional_cuts):
            prob += pulp.lpSum(x[i] for i in indices) <= limit

        prob._x_vars = x  # type: ignore[attr-defined]
        prob._own_sum_expr = own_sum_expr
        prob._own_hhi_expr = own_hhi_expr
        prob._assign = assign
        prob._ev_vars = ev_vars
        prob._pp1_vars = pp1_vars
        prob._pp1_primary_vars = pp1_primary_vars
        return prob

    # ------------------------------------------------------------------
    def _solve_one(
        self,
        plan: ShapePlan,
        projections: Dict[int, float],
        cuts: List[Tuple[List[int], int]],
    ) -> Optional[LineupResult]:
        prob = self._build_problem(plan, projections, cuts)
        if prob is None:
            return None
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if status != pulp.LpStatusOptimal:
            LOG.debug("Solver status %s for plan %s", pulp.LpStatus[status], plan.counts)
            return None

        x = prob._x_vars  # type: ignore[attr-defined]
        picked = [i for i, var in x.items() if var.value() and var.value() > 0.5]
        if len(picked) != 9:
            return None

        slots = self._assign_slots(picked)
        salary = int(self.pool.loc[picked, "Salary"].sum())
        proj = float(sum(projections.get(i, 0.0) for i in picked))
        proj_base = float(self.pool.loc[picked, "ProjBase"].sum())
        own_sum = float(prob._own_sum_expr.value()) if hasattr(prob, "_own_sum_expr") else 0.0
        own_hhi = float(prob._own_hhi_expr.value()) if hasattr(prob, "_own_hhi_expr") else 0.0

        assign = getattr(prob, "_assign", {})
        primary_team = None
        if assign:
            for (team, slot), var in assign.items():
                if slot == 0 and var.value() and var.value() > 0.5:
                    primary_team = team
                    break

        ev_size, pp1_size = self._lineup_stack_sizes(picked)
        stack_details = {
            "primary_team": primary_team,
            "ev_cluster": ev_size,
            "pp1_cluster": pp1_size,
        }
        return LineupResult(
            indices=picked,
            slots=slots,
            salary=salary,
            projection=proj,
            projection_base=proj_base,
            ownership_sum=own_sum,
            ownership_hhi=own_hhi,
            stack_details=stack_details,
        )

    # ------------------------------------------------------------------
    def _assign_slots(self, indices: List[int]) -> Dict[str, int]:
        pool = self.pool
        slots: Dict[str, int] = {}
        remaining = set(indices)

        def take(position: str, slot_names: List[str]) -> None:
            candidates = [i for i in remaining if pool.loc[i, "Pos"] == position]
            candidates.sort(
                key=lambda j: (
                    float(pool.loc[j, "ProjBase"]),
                    int(pool.loc[j, "Salary"]),
                ),
                reverse=True,
            )
            for slot in slot_names:
                if not candidates:
                    raise ValueError(f"Insufficient {position} players for slot {slot}")
                pick = candidates.pop(0)
                slots[slot] = pick
                remaining.remove(pick)

        take("G", ["G"])
        take("C", ["C1", "C2"])
        take("W", ["W1", "W2", "W3"])
        take("D", ["D1", "D2"])

        util_candidates = [i for i in remaining if self.pool.loc[i, "Pos"] in {"C", "W", "D"}]
        if len(util_candidates) != 1:
            util_candidates.sort(
                key=lambda j: (
                    float(pool.loc[j, "ProjBase"]),
                    int(pool.loc[j, "Salary"]),
                ),
                reverse=True,
            )
        if not util_candidates:
            raise ValueError("Missing UTIL candidate")
        slots["UTIL"] = util_candidates[0]
        return slots

    # ------------------------------------------------------------------
    def _lineup_stack_sizes(self, indices: List[int]) -> Tuple[int, int]:
        subset = self.pool.loc[indices]
        ev_groups = subset[subset["full_line_group"].notna()].groupby(["Team", "Full"]).size()
        pp1_groups = subset[(subset["pp_unit"] == 1)].groupby("Team").size()
        ev_max = int(ev_groups.max()) if not ev_groups.empty else 0
        pp1_max = int(pp1_groups.max()) if not pp1_groups.empty else 0
        return ev_max, pp1_max

    # ------------------------------------------------------------------
    def build_lineups(
        self,
        num_lineups: int,
        *,
        projection_guard: Optional[float],
        require_unique: bool,
        max_duplication: Optional[int],
    ) -> List[LineupResult]:
        cuts: List[Tuple[List[int], int]] = []
        seen: Dict[str, int] = {}
        results: List[LineupResult] = []

        if self.best_projection is None:
            self.best_projection = self._compute_best_projection()
            LOG.info("Best feasible projection: %.2f", self.best_projection)

        max_attempts = max(200, num_lineups * 20)
        attempts = 0
        while len(results) < num_lineups and attempts < max_attempts:
            attempts += 1
            projections = self._jitter_projections()
            lineup = None
            for plan in self.shapes:
                lineup = self._solve_one(plan, projections, cuts)
                if lineup:
                    break
            if not lineup:
                continue

            signature = lineup.signature(self.pool["Name"].tolist())
            proj_gap = (self.best_projection or lineup.projection_base) - lineup.projection_base
            if projection_guard is not None and proj_gap > projection_guard + 1e-6:
                cuts.append((lineup.indices, len(lineup.indices) - 1))
                continue
            if require_unique and signature in seen:
                cuts.append((lineup.indices, len(lineup.indices) - 1))
                continue
            if max_duplication is not None and seen.get(signature, 0) >= max_duplication:
                cuts.append((lineup.indices, len(lineup.indices) - 1))
                continue

            seen[signature] = seen.get(signature, 0) + 1
            results.append(lineup)
            cuts.append((lineup.indices, len(lineup.indices) - 1))

        if len(results) < num_lineups:
            LOG.warning("Generated %d/%d lineups before reaching attempt cap", len(results), num_lineups)
        return results

    # ------------------------------------------------------------------
    def _jitter_projections(self) -> Dict[int, float]:
        eps = float(self.tie_break_random_pct)
        if eps <= 0:
            return {i: float(self.pool.loc[i, "ProjBase"]) for i in self.pool.index}
        proj = {}
        for i in self.pool.index:
            base = float(self.pool.loc[i, "ProjBase"])
            jitter = random.uniform(1.0 - eps, 1.0 + eps)
            proj[i] = base * jitter
        return proj

    # ------------------------------------------------------------------
    def _compute_best_projection(self) -> float:
        projections = {i: float(self.pool.loc[i, "ProjBase"]) for i in self.pool.index}
        cuts: List[Tuple[List[int], int]] = []
        best = 0.0
        for plan in self.shapes:
            lineup = self._solve_one(plan, projections, cuts)
            if lineup:
                best = max(best, lineup.projection_base)
                cuts.append((lineup.indices, len(lineup.indices) - 1))
        return best


# ------------------------------------------------------------------------------
# Utilities for reporting / exports
# ------------------------------------------------------------------------------

def lineup_to_row(pool: pd.DataFrame, lineup: LineupResult) -> Dict[str, object]:
    slots = {}
    for slot, idx in lineup.slots.items():
        row = pool.loc[idx]
        slots[f"{slot}"] = row["Name"]
        slots[f"{slot}_ID"] = idx
    slots["TotalSalary"] = lineup.salary
    slots["Proj"] = round(lineup.projection_base, 3)
    slots["ProjJitter"] = round(lineup.projection, 3)
    slots["OwnershipSum"] = round(lineup.ownership_sum, 3)
    slots["OwnershipHHI"] = round(lineup.ownership_hhi, 4)
    slots.update({
        "EVCluster": lineup.stack_details.get("ev_cluster", 0),
        "PP1Cluster": lineup.stack_details.get("pp1_cluster", 0),
        "PrimaryTeam": lineup.stack_details.get("primary_team"),
    })
    return slots


# ------------------------------------------------------------------------------
# CLI glue
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("NHL optimizer with edges config")
    ap.add_argument("--date", required=True)
    ap.add_argument("--labs-dir", default="dk_data")
    ap.add_argument("--out", default="out/lineups_{date}.csv")
    ap.add_argument("--export-raw", action="store_true")
    ap.add_argument("--num-lineups", type=int, default=150)
    ap.add_argument("--min-salary", type=int, default=48000)
    ap.add_argument("--max-salary", type=int, default=DK_SALARY_CAP)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--shape", type=str, default=None, help="Override team shape (e.g. 4-3-1)")
    ap.add_argument("--pp1-full", action="store_true", help="Enable PP1 full-unit primary template")
    return ap.parse_args()


def _build_shapes(cfg: Dict, override: Optional[List[int]], enable_pp1_full: bool) -> List[ShapePlan]:
    base_shapes = cfg.get("stacks", {}).get("preferred_team_shapes", [])
    plans: List[ShapePlan] = []
    if override:
        plans.append(ShapePlan(counts=override, primary_mode="ev"))
    else:
        for shp in base_shapes:
            if isinstance(shp, str):
                parts = _parse_shape_arg(shp)
            else:
                parts = [int(x) for x in shp]
            if sum(parts) != 8:
                LOG.warning("Shape %s does not sum to 8 skaters; skipping", shp)
                continue
            plans.append(ShapePlan(counts=parts, primary_mode="ev"))
    stacks_cfg = cfg.get("stacks", {})
    if enable_pp1_full or stacks_cfg.get("pp1_full_unit", {}).get("enabled"):
        extra_shapes = override and [override] or base_shapes
        for shp in extra_shapes:
            if isinstance(shp, str):
                parts = _parse_shape_arg(shp)
            else:
                parts = [int(x) for x in shp]
            if sum(parts) != 8:
                continue
            plans.append(ShapePlan(counts=parts, primary_mode="pp1"))
    if not plans and override:
        plans.append(ShapePlan(counts=override, primary_mode="ev"))
    return plans


def main() -> None:
    args = parse_args()
    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    cfg = _load_config(args.config)
    date = args.date
    labs_dir = args.labs_dir

    ownership_cfg = cfg.get("ownership", {})
    pool = load_player_pool(labs_dir, date, ownership_cfg)
    if pool.empty:
        raise SystemExit("Empty player pool — check Labs files.")

    salary_cfg = cfg.get("salary", {})
    min_spend = int(salary_cfg.get("min_spend", args.min_salary))
    max_left = int(salary_cfg.get("max_leftover", DK_SALARY_CAP - args.min_salary))
    min_salary = max(min_spend, DK_SALARY_CAP - max_left, args.min_salary)
    max_salary = min(DK_SALARY_CAP, args.max_salary)

    goalie_cfg = cfg.get("goalie_rules", {})
    bringback_cfg = cfg.get("bringbacks", {})
    div_cfg = cfg.get("diversity", {})
    proj_guard_cfg = cfg.get("projection_guards", {})

    shapes = _build_shapes(cfg, _parse_shape_arg(args.shape), args.pp1_full)
    if not shapes:
        raise SystemExit("No feasible shapes configured.")

    tie_break = float(div_cfg.get("tie_break_random_pct", 0.1))
    projection_guard = proj_guard_cfg.get("max_proj_gap_to_pool_max")

    ownership_limits = {}
    own_cfg = cfg.get("ownership", {})
    if own_cfg.get("enable"):
        if own_cfg.get("max_sum") is not None:
            ownership_limits["max_sum"] = float(own_cfg["max_sum"])
        if own_cfg.get("max_hhi") is not None:
            ownership_limits["max_hhi"] = float(own_cfg["max_hhi"])

    optimizer = EdgesOptimizer(
        pool,
        cfg,
        shapes,
        tie_break_random_pct=tie_break,
        bringbacks_enabled=bool(bringback_cfg.get("enabled", True)),
        max_from_opponent=int(bringback_cfg.get("max_from_opponent", 0)),
        forbid_goalie_conflict=bool(goalie_cfg.get("forbid_opp_skaters", True)),
        salary_min=min_salary,
        salary_max=max_salary,
        ownership_limits=ownership_limits,
    )

    lineups = optimizer.build_lineups(
        int(cfg.get("num_lineups", args.num_lineups)),
        projection_guard=float(projection_guard) if projection_guard is not None else None,
        require_unique=bool(div_cfg.get("require_unique_lineups", True)),
        max_duplication=int(div_cfg.get("max_duplication_per_contest", 3)) if div_cfg.get("max_duplication_per_contest") is not None else None,
    )
    if not lineups:
        raise SystemExit("No lineups generated with current constraints.")

    rows = [lineup_to_row(optimizer.pool, lu) for lu in lineups]
    out_path = (args.out or "out/lineups_{date}.csv").format(date=date)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.insert(0, "LineupID", range(1, len(df) + 1))
    df.to_csv(out_path, index=False)
    LOG.info("Wrote %s", out_path)

    if args.export_raw:
        raw_path = out_path.replace(".csv", "_raw.csv")
        raw = pd.DataFrame(
            [
                {slot: optimizer.pool.loc[idx, "Name"] for slot, idx in lu.slots.items()}
                for lu in lineups
            ]
        )
        raw.insert(0, "LineupID", range(1, len(raw) + 1))
        raw.to_csv(raw_path, index=False)
        LOG.info("Wrote %s", raw_path)

    export_cfg = cfg.get("export", {}).get("dk_upload", {})
    if export_cfg.get("enable", True):
        pid_path = export_cfg.get("player_ids_path", os.path.join("dk_data", "player_ids.csv"))
        upload_path = out_path.replace(".csv", "_dk_upload.csv")
        try:
            export_dk_upload(lineups, optimizer.pool, pid_path, upload_path)
        except Exception as exc:  # pragma: no cover - export optional
            LOG.warning("DK upload export failed: %s", exc)


if __name__ == "__main__":
    main()
