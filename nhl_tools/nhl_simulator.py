#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DraftKings NHL GPP simulator with NFL-style interface and outputs."""
from __future__ import annotations

import argparse
import collections
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nhl_tools.nhl_data import (
    DK_SALARY_CAP,
    apply_external_ownership,
    load_player_reference,
    normalize_lineup_player,
    normalize_name,
)
from nhl_tools.nhl_optimizer import _load_config

LOG = logging.getLogger("nhl_sim")
DEFAULT_NOISE = 0.75  # scoring jitter per entry to break ties

# Canonical DraftKings slot order (used for output formatting)
DK_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL", "UTIL2"]


@dataclass
class PlayerRecord:
    name: str
    position: str
    team: str
    opp: str | None = None
    salary: float | None = None
    projection: float | None = None
    ceiling: float | None = None
    actual: float | None = None
    ownership: float | None = None
    full: str | None = None
    pp_unit: Optional[int] = None
    player_id: str | None = None

    @property
    def is_skater(self) -> bool:
        return self.position not in {"G", "GOALIE"}

    @property
    def is_goalie(self) -> bool:
        return self.position in {"G", "GOALIE"}


@dataclass
class Lineup:
    idx: int
    lineup_id: str
    slot_players: Dict[str, PlayerRecord]
    signature: str
    metrics: Dict[str, float]
    base_score: float
    stack_tags: Dict[str, str] = field(default_factory=dict)

    def players(self) -> Iterable[PlayerRecord]:
        return self.slot_players.values()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edges-aware NHL GPP simulator")
    parser.add_argument("--site", default="DK", help="Site (DraftKings supported)")
    parser.add_argument("--field-size", type=int, default=20000, help="Contest field size")
    parser.add_argument("--iterations", type=int, default=5000, help="Simulation iterations")
    parser.add_argument("--lineups", required=True, help="CSV of generated lineups")
    parser.add_argument(
        "--players",
        help="Optional player pool CSV (FantasyLabs export / optimizer pool)",
    )
    parser.add_argument(
        "--ownership-file",
        help="Optional ownership CSV to merge with players",
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
        help="YAML config with stack / ownership tuning",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs",
    )
    return parser


def _coerce_namespace(args: argparse.Namespace | Sequence[str] | None) -> argparse.Namespace:
    if isinstance(args, argparse.Namespace):
        return args
    parser = _build_parser()
    if args is None:
        return parser.parse_args()
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _read_lineups_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Lineup file is empty")
    return df


def _detect_slot_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map dataframe columns to canonical DK slots."""

    slot_counts = collections.Counter()
    mapping: Dict[str, str] = {}

    def _slot_type(col: str) -> Optional[str]:
        base = "".join(ch for ch in str(col).upper() if ch.isalnum())
        if not base:
            return None
        if base.startswith("UTIL"):
            return "UTIL"
        if base.startswith("GOALIE") or base == "G" or base.startswith("G"):  # goalie
            return "G"
        if base.startswith("CENTER") or base.startswith("C"):
            return "C"
        if base.startswith("WING") or base.startswith("W"):
            return "W"
        if base.startswith("DEF") or base.startswith("D"):
            return "D"
        return None

    slot_candidates: Dict[str, List[Tuple[int, str, int]]] = collections.defaultdict(list)

    def _slot_score(col: str) -> int:
        """Heuristic score favouring player-name columns over metadata."""

        lower = str(col).lower()
        score = 0
        if any(token in lower for token in ["name", "player"]):
            score += 10
        if lower.strip() in {"c1", "c2", "w1", "w2", "w3", "d1", "d2", "g", "util"}:
            score += 5
        # Penalise obvious metadata/ID columns that share the same prefix.
        if any(
            token in lower
            for token in [
                "id",
                "proj",
                "projection",
                "salary",
                "own",
                "fpts",
                "points",
                "exposure",
                "count",
                "weight",
                "team",
                "opp",
                "stack",
                "type",
                "line",
                "full",
                "pp",
                "slot",
                "position",
            ]
        ):
            score -= 8
        return score

    for order, col in enumerate(df.columns):
        slot = _slot_type(col)
        if not slot:
            continue
        score = _slot_score(col)
        slot_candidates[slot].append((order, col, score))

    for slot, candidates in slot_candidates.items():
        # Sort by our heuristic (higher score wins) while keeping the original order
        # for stability when scores tie.
        ranked = sorted(candidates, key=lambda tup: (-tup[2], tup[0]))
        slot_counts[slot] = len(ranked)
        for idx, (_, col, _) in enumerate(ranked, start=1):
            if slot == "C":
                canon = f"C{idx}"
            elif slot == "W":
                canon = f"W{idx}"
            elif slot == "D":
                canon = f"D{idx}"
            elif slot == "UTIL":
                canon = "UTIL" if "UTIL" not in mapping else f"UTIL{idx}"
            else:
                canon = "G"
            if canon in mapping:
                continue
            mapping[canon] = col

    required = {"C1", "C2", "W1", "W2", "W3", "D1", "D2", "G"}
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(
            "Lineup file missing required slots. Expected DraftKings columns "
            f"covering {sorted(required)}; inferred mapping was {mapping}."
        )
    if "UTIL" not in mapping:
        # allow single util slot but not fatal; create placeholder for order
        mapping["UTIL"] = None
    return mapping


def _extract_slot_value(row: pd.Series, column: Optional[str]) -> Optional[str]:
    if column is None:
        return None
    value = row.get(column)
    if pd.isna(value):
        return None
    return str(value).strip()


def _extract_slot_metadata(row: pd.Series, column: Optional[str]) -> Dict[str, str]:
    """Gather extra metadata columns that share the same prefix as the slot."""
    if column is None:
        return {}
    base = str(column)
    meta: Dict[str, str] = {}
    for col, value in row.items():
        if col == column:
            continue
        if not isinstance(col, str):
            continue
        if col.startswith(base + " "):
            key = col[len(base) + 1 :].strip().lower()
            meta[key] = value
    return meta


def _lineup_weights(df: pd.DataFrame) -> np.ndarray:
    for cand in ["Weight", "Weights", "Exposure", "Exposure%", "Entries", "Count"]:
        if cand in df.columns:
            weights = pd.to_numeric(df[cand], errors="coerce").fillna(0).to_numpy()
            if weights.sum() <= 0:
                continue
            if cand.lower().endswith("%") or weights.max() <= 1.0001:
                weights = weights.astype(float)
                if weights.max() > 1.0001:
                    weights = weights / 100.0
            return weights.astype(float)
    return np.ones(len(df), dtype=float)


def _build_lineups(
    df: pd.DataFrame,
    player_lookup: Dict[str, List[pd.Series]],
    ownership_df: Optional[pd.DataFrame],
    quiet: bool,
) -> Tuple[List[Lineup], Dict[str, int]]:
    mapping = _detect_slot_columns(df)
    weights = _lineup_weights(df)

    ownership_df = ownership_df.copy() if ownership_df is not None else None
    if ownership_df is not None:
        ownership_df["NameNorm"] = ownership_df["Name"].map(normalize_name)
        ownership_df["Team"] = ownership_df["Team"].astype(str).str.upper().str.strip()
        ownership_df["Pos"] = ownership_df["Pos"].astype(str).str.upper().str.strip()

    lineups: List[Lineup] = []
    dup_counter: Dict[str, int] = collections.Counter()

    for idx, row in df.iterrows():
        slot_players: Dict[str, PlayerRecord] = {}
        identifiers: List[str] = []
        for slot in DK_SLOTS:
            source_col = mapping.get(slot)
            value = _extract_slot_value(row, source_col)
            if not value:
                continue
            meta = _extract_slot_metadata(row, source_col)
            raw_player = normalize_lineup_player(value, meta, player_lookup)
            player = PlayerRecord(**raw_player)
            if ownership_df is not None and not player.ownership:
                match = ownership_df[
                    (ownership_df["NameNorm"] == normalize_name(player.name))
                    & ((ownership_df["Team"] == player.team) | ownership_df["Team"].isna())
                ]
                if not match.empty:
                    cand = match.iloc[0]
                    try:
                        player.ownership = float(cand.get("Ownership"))
                    except Exception:
                        player.ownership = None
            slot_players[slot] = player
            identifier = player.player_id or f"{normalize_name(player.name)}:{player.team}:{player.position}"
            identifiers.append(identifier)

        if not slot_players:
            continue
        signature = "|".join(sorted(identifiers))
        dup_counter[signature] += 1

        lineup_id = str(row.get("LineupID", row.get("id", idx + 1)))
        lineup = Lineup(
            idx=idx,
            lineup_id=lineup_id,
            slot_players=slot_players,
            signature=signature,
            metrics={},
            base_score=0.0,
            stack_tags={},
        )
        lineup.metrics["weight"] = float(weights[idx]) if idx < len(weights) else 1.0
        lineups.append(lineup)

    if not lineups:
        raise ValueError("No valid lineups found in lineup file")

    if not quiet:
        dup_sizes = [c for c in dup_counter.values() if c > 1]
        if dup_sizes:
            LOG.info("Detected %d duplicated lineup signatures (max dup %d)", len(dup_sizes), max(dup_sizes))
        else:
            LOG.info("All lineups unique by signature")

    return lineups, dup_counter


# ---------------------------------------------------------------------------
# Metrics + scoring
# ---------------------------------------------------------------------------


def _primary_team(players: Iterable[PlayerRecord]) -> Tuple[str, int, float]:
    team_counts: Dict[str, List[PlayerRecord]] = collections.defaultdict(list)
    for player in players:
        if player.is_skater and player.team:
            team_counts[player.team].append(player)
    if not team_counts:
        return "", 0, 0.0
    best_team = ""
    best_size = -1
    best_proj = -1.0
    for team, members in team_counts.items():
        size = len(members)
        proj = sum(float(p.projection or 0.0) for p in members)
        if size > best_size or (size == best_size and proj > best_proj):
            best_team = team
            best_size = size
            best_proj = proj
    return best_team, best_size, best_proj


def _group_counts(players: Iterable[PlayerRecord], key_fn) -> Dict[str, List[PlayerRecord]]:
    buckets: Dict[str, List[PlayerRecord]] = collections.defaultdict(list)
    for player in players:
        key = key_fn(player)
        if key:
            buckets[key].append(player)
    return buckets


def _compute_lineup_metrics(lineup: Lineup) -> None:
    players = list(lineup.players())
    salary = sum(float(p.salary or 0.0) for p in players)
    projection = sum(float(p.projection or 0.0) for p in players)
    actual = sum(float(p.actual or 0.0) for p in players if p.actual is not None)
    ceiling = sum(float(p.ceiling or 0.0) for p in players if p.ceiling is not None)

    own_vals = [float(p.ownership) for p in players if p.ownership not in (None, np.nan)]
    own_sum = float(sum(own_vals)) if own_vals else float("nan")
    own_hhi = float(sum((val / 100.0) ** 2 for val in own_vals)) if own_vals else float("nan")

    goalie_team = next((p.team for p in players if p.is_goalie), "")
    goalie_conflict = sum(1 for p in players if p.is_skater and p.opp and p.opp == goalie_team)

    primary_team, _, _ = _primary_team(players)
    opponent_candidates = {p.opp for p in players if p.team == primary_team and p.opp}
    primary_opp = next(iter(opponent_candidates)) if opponent_candidates else ""
    bringback_count = sum(1 for p in players if p.is_skater and p.team == primary_opp)

    ev_groups = _group_counts(
        (p for p in players if p.is_skater),
        lambda p: f"{p.team}:{p.full}" if p.team and p.full else None,
    )
    ev_cluster_max = max((len(v) for v in ev_groups.values()), default=0)

    pp_groups = _group_counts(
        (p for p in players if p.is_skater and p.pp_unit == 1),
        lambda p: p.team,
    )
    pp1_cluster = max((len(v) for v in pp_groups.values()), default=0)

    skater_team_counts = collections.Counter(p.team for p in players if p.is_skater and p.team)
    shape_counts = sorted((cnt for cnt in skater_team_counts.values() if cnt > 0), reverse=True)
    lineup_shape = "-".join(str(cnt) for cnt in shape_counts) if shape_counts else ""

    ev_stack_strings = [f"EV:{team}:{len(members)}" for team, members in sorted(ev_groups.items(), key=lambda kv: len(kv[1]), reverse=True) if members]
    pp_stack_strings = [f"PP1:{team}:{len(members)}" for team, members in sorted(pp_groups.items(), key=lambda kv: len(kv[1]), reverse=True) if members]

    lineup.metrics.update(
        {
            "salary": salary,
            "projection": projection,
            "actual": actual if actual else float("nan"),
            "ceiling": ceiling if ceiling else float("nan"),
            "salary_leftover": DK_SALARY_CAP - salary,
            "ownership_sum": own_sum,
            "ownership_hhi": own_hhi,
            "goalie_conflict": float(goalie_conflict),
            "bringback_count": float(bringback_count),
            "ev_cluster": float(ev_cluster_max),
            "pp1_cluster": float(pp1_cluster),
            "primary_team": primary_team,
            "primary_opponent": primary_opp,
            "lineup_shape": lineup_shape,
        }
    )

    lineup.stack_tags = {
        "Stack1 Type": ev_stack_strings[0] if ev_stack_strings else "",
        "Stack2 Type": pp_stack_strings[0] if pp_stack_strings else "",
        "Lineup Type": lineup_shape,
    }
    lineup.metrics["ev_stacks"] = [(team, members[0].full, len(members)) for team, members in ev_groups.items() if members]
    lineup.metrics["pp1_stacks"] = [(team, len(members)) for team, members in pp_groups.items() if members]


def _ownership_penalties(metrics: Dict[str, float], cfg: Dict[str, object]) -> float:
    own_cfg = cfg.get("ownership", {}) if isinstance(cfg, dict) else {}
    if not own_cfg or not own_cfg.get("enable"):
        return 0.0
    penalty = 0.0
    own_sum = metrics.get("ownership_sum")
    own_hhi = metrics.get("ownership_hhi")
    max_sum = own_cfg.get("max_sum")
    max_hhi = own_cfg.get("max_hhi")
    if isinstance(own_sum, float) and not math.isnan(own_sum) and max_sum is not None and own_sum > max_sum:
        penalty += own_cfg.get("sum_penalty", 1.0)
    if isinstance(own_hhi, float) and not math.isnan(own_hhi) and max_hhi is not None and own_hhi > max_hhi:
        penalty += own_cfg.get("hhi_penalty", 0.5)
    return penalty


def _adjusted_score(metrics: Dict[str, float], cfg: Dict[str, object]) -> float:
    score = float(metrics.get("projection", 0.0))
    score += 0.5 * max(0.0, metrics.get("ev_cluster", 0.0) - 2.0)
    score += 0.4 * max(0.0, metrics.get("pp1_cluster", 0.0) - 2.0)
    score -= 3.0 * metrics.get("goalie_conflict", 0.0)
    score -= 1.0 * min(2.0, metrics.get("bringback_count", 0.0))
    score -= 0.1 * max(0.0, metrics.get("salary_leftover", 0.0) - 500.0) / 100.0
    score -= _ownership_penalties(metrics, cfg)
    return score


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def _prepare_sim(lineups: List[Lineup], cfg: Dict[str, object]) -> None:
    for lineup in lineups:
        _compute_lineup_metrics(lineup)
        lineup.base_score = _adjusted_score(lineup.metrics, cfg)


@dataclass
class SimulationResults:
    lineup_table: pd.DataFrame
    player_table: pd.DataFrame
    stack_table: pd.DataFrame


def _run_simulation(
    lineups: List[Lineup],
    dup_counter: Dict[str, int],
    iterations: int,
    field_size: int,
    rng: np.random.Generator,
) -> SimulationResults:
    n = len(lineups)
    weights = np.array([max(lineup.metrics.get("weight", 1.0), 0.0) for lineup in lineups], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    probs = weights / weights.sum()

    base_scores = np.array([lineup.base_score for lineup in lineups], dtype=float)

    lineup_counts = np.zeros(n, dtype=np.int64)
    lineup_duplicate_counts = np.zeros(n, dtype=np.float64)
    lineup_wins = np.zeros(n, dtype=np.int64)

    total_entries = iterations * field_size

    player_meta: Dict[str, Dict[str, object]] = {}
    player_counts = collections.Counter()
    player_wins = collections.Counter()

    ev_stack_counts: Dict[Tuple[str, str, int], int] = collections.Counter()
    pp_stack_counts: Dict[Tuple[str, int], int] = collections.Counter()
    shape_counts: Dict[str, int] = collections.Counter()
    goalie_conflict_counts: Dict[str, int] = collections.Counter()

    for _ in range(iterations):
        counts = rng.multinomial(field_size, probs)
        noise = rng.normal(0.0, DEFAULT_NOISE, size=field_size)
        best_scores = np.full(n, -np.inf, dtype=float)
        idx_ptr = 0
        for lineup_idx, count in enumerate(counts):
            if count == 0:
                continue
            lineup = lineups[lineup_idx]
            lineup_counts[lineup_idx] += count
            lineup_duplicate_counts[lineup_idx] += max(0, count - 1)

            stack_weight = int(count)
            for team, full, size in lineup.metrics.get("ev_stacks", []):
                if team and full:
                    ev_stack_counts[(team, str(full), int(size))] += stack_weight
            for team, size in lineup.metrics.get("pp1_stacks", []):
                if team:
                    pp_stack_counts[(team, int(size))] += stack_weight
            shape = lineup.metrics.get("lineup_shape", "")
            if shape:
                shape_counts[shape] += stack_weight
            conflict_key = "Has Conflict" if lineup.metrics.get("goalie_conflict", 0.0) else "No Conflict"
            goalie_conflict_counts[conflict_key] += stack_weight

            slice_noise = noise[idx_ptr : idx_ptr + count]
            idx_ptr += count
            top_score = lineup.base_score + (slice_noise.max() if len(slice_noise) else 0.0)
            best_scores[lineup_idx] = top_score

            for player in lineup.players():
                key = normalize_name(player.name)
                player_counts[key] += count
                if key not in player_meta:
                    player_meta[key] = {
                        "Player": player.name,
                        "Position": player.position,
                        "Team": player.team,
                        "Fpts Act": player.actual,
                        "Proj Own%": player.ownership,
                    }

        winner_idx = int(best_scores.argmax())
        if best_scores[winner_idx] > -np.inf:
            lineup_wins[winner_idx] += 1
            winning_lineup = lineups[winner_idx]
            for player in winning_lineup.players():
                player_wins[normalize_name(player.name)] += 1

    # Build lineup table
    lineup_rows: List[Dict[str, object]] = []
    total_counts = float(total_entries) if total_entries else 1.0
    for idx, lineup in enumerate(lineups):
        slots = {slot: "" for slot in DK_SLOTS}
        for slot, player in lineup.slot_players.items():
            slots[slot] = player.name
        win_pct = (lineup_wins[idx] / iterations * 100.0) if iterations else 0.0
        sim_own_pct = lineup_counts[idx] / total_counts * 100.0 if total_counts else 0.0
        sim_dupes = lineup_duplicate_counts[idx] / iterations if iterations else 0.0
        row = {
            "Lineup": lineup.lineup_id,
            "Win%": win_pct,
            "Top1%": win_pct,
            "Avg Return": float("nan"),
            "Fpts Proj": lineup.metrics.get("projection"),
            "Field Fpts Proj": lineup.base_score,
            "Fpts Act": lineup.metrics.get("actual"),
            "Ceiling": lineup.metrics.get("ceiling"),
            "Salary": lineup.metrics.get("salary"),
            "Salary Leftover": lineup.metrics.get("salary_leftover"),
            "EV Cluster": lineup.metrics.get("ev_cluster"),
            "PP1 Cluster": lineup.metrics.get("pp1_cluster"),
            "BringBacks": lineup.metrics.get("bringback_count"),
            "Skaters vs Goalie": lineup.metrics.get("goalie_conflict"),
            "Own Sum": lineup.metrics.get("ownership_sum"),
            "HHI": lineup.metrics.get("ownership_hhi"),
            "Lineup Dupes": dup_counter.get(lineup.signature, 1),
            "Sim Dupes": sim_dupes,
            "Sim Own%": sim_own_pct,
        }
        row.update(slots)
        row.update(lineup.stack_tags)
        lineup_rows.append(row)

    lineup_table = pd.DataFrame(lineup_rows)
    ordered_cols = (
        DK_SLOTS
        + [
            "Lineup",
            "Fpts Proj",
            "Field Fpts Proj",
            "Fpts Act",
            "Ceiling",
            "Salary",
            "Salary Leftover",
            "EV Cluster",
            "PP1 Cluster",
            "BringBacks",
            "Skaters vs Goalie",
            "Own Sum",
            "HHI",
            "Lineup Dupes",
            "Sim Dupes",
            "Sim Own%",
            "Win%",
            "Top1%",
            "Avg Return",
            "Stack1 Type",
            "Stack2 Type",
            "Lineup Type",
        ]
    )
    lineup_table = lineup_table[[c for c in ordered_cols if c in lineup_table.columns]]

    # Player exposures
    player_rows: List[Dict[str, object]] = []
    for key, meta in sorted(player_meta.items(), key=lambda kv: kv[1]["Player"].lower()):
        count = player_counts.get(key, 0)
        wins = player_wins.get(key, 0)
        sim_pct = count / total_counts * 100.0 if total_counts else 0.0
        win_pct = wins / iterations * 100.0 if iterations else 0.0
        player_rows.append(
            {
                "Player": meta.get("Player"),
                "Position": meta.get("Position"),
                "Team": meta.get("Team"),
                "Fpts Act": meta.get("Fpts Act"),
                "Win%": win_pct,
                "Top1%": win_pct,
                "Sim. Own%": sim_pct,
                "Proj. Own%": meta.get("Proj Own%"),
                "Avg. Return": float("nan"),
            }
        )

    player_table = pd.DataFrame(player_rows)

    # Stack exposures
    stack_rows: List[Dict[str, object]] = []
    if ev_stack_counts:
        for (team, ev_line, size), count in sorted(ev_stack_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            stack_rows.append(
                {
                    "Category": "EV Stack",
                    "Team": team,
                    "Descriptor": ev_line,
                    "Size": size,
                    "Sim%": count / total_entries * 100.0 if total_entries else 0.0,
                }
            )
    if pp_stack_counts:
        for (team, size), count in sorted(pp_stack_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            stack_rows.append(
                {
                    "Category": "PP1 Stack",
                    "Team": team,
                    "Descriptor": "PP1",
                    "Size": size,
                    "Sim%": count / total_entries * 100.0 if total_entries else 0.0,
                }
            )
    if shape_counts:
        for shape, count in sorted(shape_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            stack_rows.append(
                {
                    "Category": "Lineup Shape",
                    "Team": "",
                    "Descriptor": shape,
                    "Size": np.nan,
                    "Sim%": count / total_entries * 100.0 if total_entries else 0.0,
                }
            )
    if goalie_conflict_counts:
        for descriptor, count in sorted(goalie_conflict_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            stack_rows.append(
                {
                    "Category": "Skaters vs Goalie",
                    "Team": "",
                    "Descriptor": descriptor,
                    "Size": np.nan,
                    "Sim%": count / total_entries * 100.0 if total_entries else 0.0,
                }
            )

    stack_table = pd.DataFrame(stack_rows)
    return SimulationResults(lineup_table=lineup_table, player_table=player_table, stack_table=stack_table)


# ---------------------------------------------------------------------------
# Output + CLI entry point
# ---------------------------------------------------------------------------


def _ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_outputs(
    results: SimulationResults,
    site: str,
    outdir: str,
    field_size: int,
    iterations: int,
) -> Tuple[str, str, str]:
    prefix = f"{site.upper()}_gpp_sim"
    lineup_path = os.path.join(outdir, f"{prefix}_lineups_{field_size}_{iterations}.csv")
    player_path = os.path.join(outdir, f"{prefix}_player_exposure_{field_size}_{iterations}.csv")
    stack_path = os.path.join(outdir, f"{prefix}_stack_exposure_{field_size}_{iterations}.csv")

    results.lineup_table.to_csv(lineup_path, index=False)
    results.player_table.to_csv(player_path, index=False)
    results.stack_table.to_csv(stack_path, index=False)
    return lineup_path, player_path, stack_path


def _configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    LOG.setLevel(level)


def main(args: argparse.Namespace | Sequence[str] | None = None) -> None:
    ns = _coerce_namespace(args)
    _configure_logging(ns.quiet)

    if ns.site.upper() != "DK":
        raise SystemExit("Only DraftKings (DK) simulations are supported at this time")

    cfg = _load_config(ns.config) if ns.config else {}

    player_lookup: Dict[str, List[pd.Series]] = {}
    ownership_df: Optional[pd.DataFrame] = None
    if ns.players:
        players_df = load_player_reference(ns.players)
        if ns.ownership_file:
            players_df = apply_external_ownership(players_df, ns.ownership_file, None)
        player_lookup = collections.defaultdict(list)
        for _, row in players_df.iterrows():
            player_lookup[normalize_name(row["Name"])].append(row)
        if "Ownership" in players_df.columns:
            ownership_df = players_df[["Name", "Team", "Pos", "Ownership"]].copy()
    elif ns.ownership_file:
        ownership_df = apply_external_ownership(pd.DataFrame(), ns.ownership_file, None)

    if not player_lookup and not ns.quiet:
        LOG.warning("No player reference data provided; metrics may be sparse")

    df = _read_lineups_csv(ns.lineups)
    lineups, dup_counter = _build_lineups(df, player_lookup, ownership_df, ns.quiet)
    _prepare_sim(lineups, cfg)

    rng = np.random.default_rng(ns.seed)
    results = _run_simulation(lineups, dup_counter, ns.iterations, ns.field_size, rng)

    outdir = _ensure_outdir(ns.outdir)
    lineup_path, player_path, stack_path = _write_outputs(
        results, ns.site, outdir, ns.field_size, ns.iterations
    )

    if not ns.quiet:
        LOG.info("Wrote lineups CSV -> %s", lineup_path)
        LOG.info("Wrote player exposure CSV -> %s", player_path)
        LOG.info("Wrote stack exposure CSV -> %s", stack_path)


if __name__ == "__main__":
    main()
