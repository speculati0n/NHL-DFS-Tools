from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .nhl_data import (
    normalize_name,
    load_player_reference_for_date,
)

# --- Configuration defaults (can be exposed to YAML later) ---
DEFAULT_NOISE = 5.0  # per-entry gaussian noise to create variance in the simulated field
DK_SLOTS = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL", "UTIL2"]

# ----------------------- Data classes ------------------------
@dataclass
class PlayerRecord:
    name: str
    position: str
    team: str
    opp: Optional[str]
    salary: float
    projection: float
    ceiling: float
    full: Optional[str]      # EV line tag e.g. 1F/2F/3F/1D/2D
    pp_unit: Optional[int]   # 1/2 for PP1/PP2
    ownership: Optional[float] = None
    actual: Optional[float] = None
    canonical_name: Optional[str] = None
    player_id: Optional[str] = None


@dataclass(init=False)
class Lineup:
    id: int
    slots: Dict[str, PlayerRecord]
    base_score: float = 0.0
    metrics: Dict[str, float] = None  # computed props (salary, clusters, conflicts, etc.)
    signature: Optional[str] = None
    idx: Optional[int] = None

    def __init__(
        self,
        id: Optional[int] = None,
        slots: Optional[Dict[str, PlayerRecord]] = None,
        *,
        idx: Optional[int] = None,
        lineup_id: Optional[str] = None,
        slot_players: Optional[Dict[str, PlayerRecord]] = None,
        signature: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        base_score: float = 0.0,
    ) -> None:
        if slots is None and slot_players is not None:
            slots = slot_players
        if slots is None:
            slots = {}
        if id is None and lineup_id is not None:
            try:
                id = int(lineup_id)
            except Exception:
                id = lineup_id
        if isinstance(id, (int, np.integer)):
            self.id = int(id)
        elif isinstance(id, str):
            try:
                self.id = int(id)
            except ValueError:
                self.id = id
        elif id is None:
            self.id = 0
        else:
            self.id = id
        self.slots = slots
        self.base_score = float(base_score)
        self.metrics = metrics
        self.signature = signature
        self.idx = idx
        self.lineup_id = lineup_id if lineup_id is not None else self.id

    def players(self) -> List[PlayerRecord]:
        return [self.slots[s] for s in DK_SLOTS if s in self.slots]


# ----------------------- I/O helpers -------------------------
def _detect_slot_columns(df: pd.DataFrame) -> List[str]:
    chosen: Dict[str, Tuple[int, str]] = {}
    for col in df.columns:
        cu = str(col).upper().strip()
        for slot in DK_SLOTS:
            score = None
            if cu == slot:
                score = 3
            elif cu.startswith(f"{slot} ") and ("NAME" in cu or "PLAYER" in cu):
                score = 2
            elif cu.startswith(slot):
                score = 1
            if score is not None:
                if slot not in chosen or score > chosen[slot][0]:
                    chosen[slot] = (score, col)
    ordered = []
    for slot in DK_SLOTS:
        if slot in chosen:
            ordered.append(chosen[slot][1])
    return ordered


def _read_lineups(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # allow either Lineup/LineupID column or create one
    id_col = None
    for cand in ("LineupID", "Lineup", "lineup_id", "lineup"):
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        df["Lineup"] = np.arange(1, len(df) + 1)
        id_col = "Lineup"
    df.rename(columns={id_col: "Lineup"}, inplace=True)
    return df


def _extract_slot_value(row: pd.Series, col: str) -> Tuple[str, Optional[str]]:
    v = row.get(col, "")
    pid: Optional[str] = None
    if isinstance(v, str):
        value = v.strip()
        if "(" in value and value.endswith(")"):
            name_part = value[: value.rfind("(")].strip()
            id_part = value[value.rfind("(") + 1 : -1].strip()
            if id_part:
                pid = id_part
            return name_part, pid
        return value, None
    if v in (None, ""):
        return "", pid
    return str(v), pid



# ----------------------- Player helpers ----------------------
def _player_key(player: PlayerRecord) -> str:
    """Normalized key for aggregating players across outputs."""

    source = player.canonical_name or player.name
    return normalize_name(source)


def _format_player_name(name: Optional[str], player_id: Optional[str]) -> str:
    if not name:
        return ""
    if player_id:
        return f"{name} ({player_id})"
    return name


def _format_player_cell(player: PlayerRecord) -> str:
    display = player.canonical_name or player.name
    pid = player.player_id if player.player_id else None
    return _format_player_name(display, pid)


# ----------------------- Metric builders ---------------------
def _group_counts(players: List[PlayerRecord], key_fn) -> Dict[Tuple, int]:
    counts: Dict[Tuple, int] = {}
    for p in players:
        k = key_fn(p)
        if k is None:
            continue
        counts[k] = counts.get(k, 0) + 1
    return counts


def _compute_lineup_metrics(lineup: Lineup) -> Dict[str, float]:
    players = lineup.players()

    # salary & projection/ceiling
    salary = float(sum([p.salary or 0.0 for p in players]))
    proj = float(sum([p.projection or 0.0 for p in players]))
    ceil = float(sum([p.ceiling or 0.0 for p in players]))

    # EV line clusters: same team + same 'full'
    ev_counts = _group_counts(
        players,
        key_fn=lambda p: (p.team, p.full) if (p.team and p.full) else None,
    )
    ev_cluster = max(ev_counts.values()) if ev_counts else 0
    ev_stacks = []
    for (team, full), cnt in ev_counts.items():
        if team and full and cnt >= 2:
            ev_stacks.append((team, full, cnt))

    # PP1 cluster: same team with pp_unit == 1
    pp_counts = _group_counts(
        [p for p in players if p.pp_unit == 1 and p.team],
        key_fn=lambda p: (p.team,),
    )
    pp1_cluster = max(pp_counts.values()) if pp_counts else 0
    pp1_stacks = []
    for (team,), cnt in pp_counts.items():
        if team and cnt >= 2:
            pp1_stacks.append((team, cnt))

    # goalie vs skaters conflicts
    goalie_teams = set([p.team for p in players if p.position == "G" and p.team])
    conflicts = 0
    for p in players:
        if p.position != "G" and p.opp and p.opp in goalie_teams:
            conflicts += 1

    # bring-backs: skaters from primary opponent of the largest stack team
    primary_team = None
    if ev_stacks:
        primary_team = sorted(ev_stacks, key=lambda t: t[2], reverse=True)[0][0]
    elif pp1_stacks:
        primary_team = sorted(pp1_stacks, key=lambda t: t[1], reverse=True)[0][0]
    bringbacks = 0
    if primary_team:
        opps = set([p.opp for p in players if p.team == primary_team and p.opp])
        if len(opps) == 1:
            primary_opp = list(opps)[0]
            bringbacks = sum(1 for p in players if p.team == primary_opp)

    # lineup shape (team counts)
    team_counts = {}
    for p in players:
        if p.team:
            team_counts[p.team] = team_counts.get(p.team, 0) + 1
    shape = "-".join(str(c) for c in sorted(team_counts.values(), reverse=True))

    return {
        "salary": salary,
        "projection": proj,
        "ceiling": ceil,
        "ev_cluster": ev_cluster,
        "pp1_cluster": pp1_cluster,
        "ev_stacks": ev_stacks,
        "pp1_stacks": pp1_stacks,
        "goalie_conflict": conflicts,
        "bringbacks": bringbacks,
        "lineup_shape": shape,
    }


def _adjusted_score(lineup: Lineup) -> float:
    """Base score used for field sampling: projection plus NHL edges."""
    m = lineup.metrics
    score = m.get("projection", 0.0)
    # correlation boosts
    score += 0.5 * max(0, m.get("ev_cluster", 0) - 2)     # reward 3–4 EV
    score += 0.4 * max(0, m.get("pp1_cluster", 0) - 2)    # reward 3–5 PP1
    # negative correlations
    score -= 3.0 * m.get("goalie_conflict", 0)
    score -= 1.0 * min(2, m.get("bringbacks", 0))
    # near-cap preference
    salary_left = max(0.0, 50000.0 - m.get("salary", 0.0))
    score -= 0.1 * max(0.0, salary_left - 500) / 100.0
    return float(score)


# ----------------------- Build lineups -----------------------
def _build_lineups(lineups_df: pd.DataFrame,
                   ref_df: pd.DataFrame,
                   *_unused_args,
                   **_unused_kwargs) -> List[Lineup]:
    if not isinstance(ref_df, pd.DataFrame):
        ref_df = pd.DataFrame(ref_df)
    slot_cols = _detect_slot_columns(lineups_df)
    if len(slot_cols) < 8:
        raise ValueError(f"Could not detect DK slots in lineups file; found: {slot_cols}")

    # index player reference by normalized name (keep multiple rows under same key)
    nm = "Player" if "Player" in ref_df.columns else "Name"
    name_groups: Dict[str, List[dict]] = {}
    for _, r in ref_df.iterrows():
        key = normalize_name(str(r[nm]))
        name_groups.setdefault(key, []).append(r)

    # helper to choose best ref row for lineup slot
    def choose_ref(name: str, slot_pos: str) -> Optional[dict]:
        key = normalize_name(name)
        cands = name_groups.get(key, [])
        if not cands:
            return None
        # score by Team/Pos/Full/PP presence, tie-break by higher Proj
        best = None
        best_score = -1e9
        for r in cands:
            score = 0
            poscol = "Pos" if "Pos" in r.index else "Position" if "Position" in r.index else None
            teamcol = "Team" if "Team" in r.index else "TeamAbbrev" if "TeamAbbrev" in r.index else None
            if poscol and str(r[poscol]).upper().startswith(slot_pos[:1]):
                score += 2
            if r.get("Full") not in (None, "", np.nan):
                score += 2
            if not pd.isna(r.get("PP", np.nan)):
                score += 1
            score += float(r.get("Proj", 0.0)) * 1e-6  # tiny nudge
            if score > best_score:
                best_score = score
                best = r
        return best

    lineups: List[Lineup] = []
    for i, row in lineups_df.iterrows():
        slots: Dict[str, PlayerRecord] = {}
        for col in slot_cols:
            name, name_player_id = _extract_slot_value(row, col)
            if not isinstance(name, str) or not name.strip():
                continue
            slot_pos = "G" if col.upper().startswith("G") else \
                       "D" if col.upper().startswith("D") else \
                       "C" if col.upper().startswith("C") else "W"
            col_prefix = str(col).split()[0].upper()
            cu = str(col).upper().strip()
            canonical_slot = None
            for slot in DK_SLOTS:
                if cu == slot or cu.startswith(f"{slot} "):
                    canonical_slot = slot
                    break
            if canonical_slot is None:
                canonical_slot = str(col)

            def _lookup(keyword: str) -> Optional[float | str]:
                keyword_upper = keyword.upper()
                for c in lineups_df.columns:
                    cu = str(c).upper()
                    if cu.startswith(col_prefix) and keyword_upper in cu and "NAME" not in cu:
                        return row.get(c)
                return None

            id_from_column = _normalize_player_id(_lookup("ID"))
            if id_from_column:
                name_player_id = id_from_column

            ref = choose_ref(name, slot_pos)
            if ref is None:
                # minimal fallback player
                salary_val = _lookup("SAL")
                proj_val = _lookup("PROJ")
                ceil_val = _lookup("CEIL")
                team_val = _lookup("TEAM")
                opp_val = _lookup("OPP")
                slots[canonical_slot] = PlayerRecord(
                    name=name,
                    position=slot_pos,
                    team=str(team_val).upper() if isinstance(team_val, str) else "",
                    opp=str(opp_val).upper() if isinstance(opp_val, str) else None,
                    salary=float(salary_val) if salary_val not in (None, "") else 0.0,
                    projection=float(proj_val) if proj_val not in (None, "") else 0.0,
                    ceiling=float(ceil_val) if ceil_val not in (None, "") else 0.0,
                    full=None,
                    pp_unit=None,
                    canonical_name=name,
                    player_id=name_player_id,
                )
                continue

            pos = ref["Pos"] if "Pos" in ref else ref.get("Position", slot_pos)
            team = ref["Team"] if "Team" in ref else ref.get("TeamAbbrev", "")
            opp = ref.get("Opp") if "Opp" in ref else ref.get("Opponent", None)
            salary = float(ref.get("Salary", 0) or 0)
            proj = float(ref.get("Proj", 0) or 0)
            ceil = float(ref.get("Ceiling", 0) or 0)
            full = ref.get("Full", None)
            pp = ref.get("PP", None)
            pp = int(pp) if not pd.isna(pp) else None
            canon_name = ref.get("Name", ref.get("Player", name))
            if pd.isna(canon_name):
                canon_name = name
            canon_name = str(canon_name).strip()
            player_id = ref.get("PlayerID", None)
            if pd.isna(player_id):
                player_id = None
            elif isinstance(player_id, float) and player_id.is_integer():
                player_id = str(int(player_id))
            else:
                player_id = str(player_id).strip()
                if not player_id:
                    player_id = None
            if not player_id and name_player_id:
                player_id = name_player_id

            slots[canonical_slot] = PlayerRecord(
                name=name, position=str(pos), team=str(team), opp=str(opp) if opp is not None else None,
                salary=salary, projection=proj, ceiling=ceil, full=str(full) if full not in (None, "", np.nan) else None,
                pp_unit=pp, canonical_name=canon_name, player_id=player_id
            )

        lineup_id = row.get("Lineup") if isinstance(row, pd.Series) else None
        if lineup_id is None and isinstance(row, pd.Series):
            lineup_id = row.get("LineupID")
        if lineup_id is None:
            lineup_id = i + 1
        l = Lineup(id=lineup_id, slots=slots)
        l.metrics = _compute_lineup_metrics(l)

        # salary fallback if unresolved: pull from optimizer-exported total if present
        if l.metrics.get("salary", 0.0) <= 0.0:
            for opt_col in ("TotalSalary", "Total Salary", "SalarySum", "Salary"):
                if opt_col in lineups_df.columns:
                    try:
                        l.metrics["salary"] = float(row[opt_col])
                        break
                    except Exception:
                        pass

        l.base_score = _adjusted_score(l)
        lineups.append(l)

    if _unused_args or _unused_kwargs:
        return lineups, {}
    return lineups


# legacy hook for tests / compatibility
def _prepare_sim(lineups: List[Lineup], cfg: Optional[dict] = None) -> None:
    for lineup in lineups:
        lineup.metrics = _compute_lineup_metrics(lineup)
        lineup.base_score = _adjusted_score(lineup)


# ----------------------- Simulation core ---------------------
@dataclass
class SimulationResults:
    lineup_wins: np.ndarray
    lineup_counts: np.ndarray
    lineup_dup_counts: np.ndarray
    player_wins: Dict[str, int]
    player_counts: Dict[str, int]
    ev_stack_counts: Dict[Tuple[str, str, int], int]
    pp_stack_counts: Dict[Tuple[str, int], int]
    shape_counts: Dict[str, int]
    goalie_conflict_counts: Dict[str, int]


class LegacySimulationResults:
    def __init__(
        self,
        lineups: List[Lineup],
        sim: SimulationResults,
        field_size: int,
        iterations: int,
    ) -> None:
        self.raw = sim
        self.lineup_table = _lineups_table(lineups, sim, field_size, iterations)
        self.player_table = _player_exposure_table(lineups, sim, field_size, iterations)
        self.stack_table = _stack_exposure_table(sim)


def _simulate_field(
    lineups: List[Lineup],
    field_size: int,
    iterations: int,
    rng: np.random.Generator,
) -> SimulationResults:
    n = len(lineups)
    assert n > 0

    # probabilities proportional to base_score (softmax for stability)
    raw = np.array([l.base_score for l in lineups], dtype=float)
    raw = raw - np.max(raw)
    probs = np.exp(raw)
    probs = probs / probs.sum()

    lineup_wins = np.zeros(n, dtype=int)
    lineup_counts = np.zeros(n, dtype=int)         # total times each lineup appears in the field
    lineup_dup_counts = np.zeros(n, dtype=int)     # total duplicates (count-1) accumulated

    player_wins: Dict[str, int] = {}
    player_counts: Dict[str, int] = {}

    ev_stack_counts: Dict[Tuple[str, str, int], int] = {}
    pp_stack_counts: Dict[Tuple[str, int], int] = {}
    shape_counts: Dict[str, int] = {}
    goalie_conflict_counts: Dict[str, int] = {"has_conflict": 0, "no_conflict": 0}

    for _ in range(iterations):
        counts = rng.multinomial(field_size, probs)
        noise = rng.normal(0.0, DEFAULT_NOISE, size=field_size)

        best_scores = np.full(n, -np.inf, dtype=float)
        idx_ptr = 0

        for idx, cnt in enumerate(counts):
            if cnt == 0:
                continue

            lu = lineups[idx]
            lineup_counts[idx] += cnt
            lineup_dup_counts[idx] += max(0, cnt - 1)

            # stacks / shape / conflicts accounting
            w = int(cnt)
            for team, full, size in lu.metrics.get("ev_stacks", []):
                if team and full:
                    ev_stack_counts[(team, str(full), int(size))] = ev_stack_counts.get((team, str(full), int(size)), 0) + w
            for team, size in lu.metrics.get("pp1_stacks", []):
                if team:
                    pp_stack_counts[(team, int(size))] = pp_stack_counts.get((team, int(size)), 0) + w
            shape = lu.metrics.get("lineup_shape", "")
            if shape:
                shape_counts[shape] = shape_counts.get(shape, 0) + w
            if lu.metrics.get("goalie_conflict", 0) > 0:
                goalie_conflict_counts["has_conflict"] += w
            else:
                goalie_conflict_counts["no_conflict"] += w

            # player sim ownership accounting
            for p in lu.players():
                key = _player_key(p)
                player_counts[key] = player_counts.get(key, 0) + cnt
                if key not in player_wins:
                    player_wins[key] = 0

            # sample cnt entries for this lineup and keep max score for winner selection
            entry_noise = noise[idx_ptr: idx_ptr + cnt]
            idx_ptr += cnt
            max_entry = float(np.max(lu.base_score + entry_noise))
            if max_entry > best_scores[idx]:
                best_scores[idx] = max_entry

        # choose a single winner this iteration
        winner_idx = int(best_scores.argmax())
        if best_scores[winner_idx] > -np.inf:
            lineup_wins[winner_idx] += 1
            for p in lineups[winner_idx].players():
                key = _player_key(p)
                player_wins[key] = player_wins.get(key, 0) + 1

    return SimulationResults(
        lineup_wins=lineup_wins,
        lineup_counts=lineup_counts,
        lineup_dup_counts=lineup_dup_counts,
        player_wins=player_wins,
        player_counts=player_counts,
        ev_stack_counts=ev_stack_counts,
        pp_stack_counts=pp_stack_counts,
        shape_counts=shape_counts,
        goalie_conflict_counts=goalie_conflict_counts,
    )


def _run_simulation(lineups: List[Lineup], *args):
    if len(args) == 3 and not isinstance(args[0], dict):
        field_size, iterations, rng = args
        if not isinstance(field_size, int):
            raise TypeError("field_size must be int when calling _run_simulation(lineups, field_size, iterations, rng)")
        return _simulate_field(lineups, field_size, iterations, rng)
    if len(args) == 4 and isinstance(args[0], dict):
        _, field_size, iterations, rng = args
        sim = _simulate_field(lineups, field_size, iterations, rng)
        return LegacySimulationResults(lineups, sim, field_size, iterations)
    raise TypeError("Unsupported arguments for _run_simulation")


# ----------------------- Outputs -----------------------------
def _lineups_table(
    lineups: List[Lineup],
    sim: SimulationResults,
    field_size: int,
    iterations: int,
) -> pd.DataFrame:
    rows = []
    denom_field = max(1, field_size * iterations)
    denom_iters = max(1, iterations)

    for i, lu in enumerate(lineups):
        m = lu.metrics
        row = {slot: "" for slot in DK_SLOTS}
        for s in DK_SLOTS:
            if s in lu.slots:
                row[s] = _format_player_cell(lu.slots[s])

        row.update({
            "Lineup": lu.id,
            "Fpts Proj": float(m.get("projection", 0.0)),
            "Field Fpts Proj": float(lu.base_score),
            "Fpts Act": -2,          # keep placeholder column for parity (if no actuals)
            "Ceiling": float(m.get("ceiling", 0.0)),
            "Salary": float(m.get("salary", 0.0)),
            "Salary Left": max(0.0, 50000.0 - float(m.get("salary", 0.0))),
            "EV Cluster": int(m.get("ev_cluster", 0)),
            "PP1 Cluster": int(m.get("pp1_cluster", 0)),
            "BringBacks": int(m.get("bringbacks", 0)),
            "Skaters vs Goalie": int(m.get("goalie_conflict", 0)),
            "Own Sum": np.nan,  # optional; left for future ownership integration
            "HHI": np.nan,      # optional
            "Lineup Dupes": 1,  # the input lineups are unique rows
            "Sim Dupes": int(sim.lineup_dup_counts[i]),
            "Sim Own%": 100.0 * sim.lineup_counts[i] / denom_field,
            "Win%": 100.0 * sim.lineup_wins[i] / denom_iters,
            "Top1%": 100.0 * sim.lineup_wins[i] / denom_iters,
            "Avg Return": 0.0,  # optional if we add payout model later
            "Stack1 Typ": "",
            "Stack2 Typ": "",
            "Lineup Type": m.get("lineup_shape", ""),
        })
        # labels for top stacks (if present)
        if m.get("ev_stacks"):
            t, f, sz = sorted(m["ev_stacks"], key=lambda x: x[2], reverse=True)[0]
            row["Stack1 Typ"] = f"EV:{t}:{sz}"
        if m.get("pp1_stacks"):
            t, sz = sorted(m["pp1_stacks"], key=lambda x: x[1], reverse=True)[0]
            row["Stack2 Typ"] = f"PP1:{t}:{sz}"

        rows.append(row)

    return pd.DataFrame(rows)


def _player_exposure_table(
    lineups: List[Lineup],
    sim: SimulationResults,
    field_size: int,
    iterations: int,
) -> pd.DataFrame:
    denom_field = max(1, field_size * iterations)
    denom_iters = max(1, iterations)

    counts: Dict[str, int] = sim.player_counts
    wins: Dict[str, int] = sim.player_wins

    # build roster-level meta to get Position/Team plus display info
    meta: Dict[str, Dict[str, Optional[str]]] = {}
    for lu in lineups:
        for p in lu.players():
            key = _player_key(p)
            info = meta.get(key)
            record = {
                "position": p.position,
                "team": p.team,
                "name": p.canonical_name or p.name,
                "player_id": p.player_id,
            }
            if info is None:
                meta[key] = record
            else:
                # retain first seen position/team but prefer records with IDs for naming
                if not info.get("player_id") and record.get("player_id"):
                    info.update({
                        "name": record.get("name"),
                        "player_id": record.get("player_id"),
                    })

    rows = []
    for key, cnt in counts.items():
        win = wins.get(key, 0)
        info = meta.get(key, {})
        pos = info.get("position", "")
        team = info.get("team", "")
        label = _format_player_name(info.get("name"), info.get("player_id")) or key
        rows.append({
            "Player": label,
            "Position": pos,
            "Team": team,
            "Win%": 100.0 * win / denom_iters,
            "Sim. Own%": 100.0 * cnt / denom_field,
            "Proj. Own%": np.nan,
        })

    return pd.DataFrame(rows)


def _stack_exposure_table(sim: SimulationResults) -> pd.DataFrame:
    rows = []
    for (team, full, size), cnt in sim.ev_stack_counts.items():
        rows.append({"Type": "EV", "Team": team, "Unit": str(full), "Size": int(size), "Sim%": cnt})
    for (team, size), cnt in sim.pp_stack_counts.items():
        rows.append({"Type": "PP1", "Team": team, "Unit": "1", "Size": int(size), "Sim%": cnt})
    for shape, cnt in sim.shape_counts.items():
        rows.append({"Type": "Shape", "Team": "", "Unit": "", "Size": 0, "Sim%": cnt})
    for k, cnt in sim.goalie_conflict_counts.items():
        rows.append({"Type": "VsGoalie", "Team": "", "Unit": k, "Size": 0, "Sim%": cnt})
    return pd.DataFrame(rows)


def _write_outputs(
    outdir: str,
    field_size: int,
    iterations: int,
    lineups_df: pd.DataFrame,
    players_df: pd.DataFrame,
    lineups_table: pd.DataFrame,
    player_exposure: pd.DataFrame,
    stack_exposure: pd.DataFrame,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    lineups_out = os.path.join(outdir, f"DK_gpp_sim_lineups_{field_size}_{iterations}.csv")
    players_out = os.path.join(outdir, f"DK_gpp_sim_player_exposure_{field_size}_{iterations}.csv")
    stacks_out = os.path.join(outdir, f"DK_gpp_sim_stack_exposure_{field_size}_{iterations}.csv")
    lineups_table.to_csv(lineups_out, index=False)
    player_exposure.to_csv(players_out, index=False)
    stack_exposure.to_csv(stacks_out, index=False)


# ----------------------- Entry point -------------------------
def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    lineups_df = _read_lineups(args.lineups)
    players_df = load_player_reference_for_date(args.players, args.date)

    lineups = _build_lineups(lineups_df, players_df)
    sim = _run_simulation(lineups, args.field_size, args.iterations, rng)

    lineups_table = _lineups_table(lineups, sim, args.field_size, args.iterations)
    player_exposure = _player_exposure_table(lineups, sim, args.field_size, args.iterations)
    stack_exposure = _stack_exposure_table(sim)

    _write_outputs(args.outdir, args.field_size, args.iterations,
                   lineups_df, players_df, lineups_table, player_exposure, stack_exposure)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="DK")
    p.add_argument("--field-size", type=int, required=True)
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--lineups", required=True, help="optimizer output CSV of unique lineups")
    p.add_argument("--players", required=True, help="dir or CSV of player refs (Labs)")
    p.add_argument("--date", default=None, help="slate date YYYY-MM-DD for filtering player refs")
    p.add_argument("--outdir", default="output")
    p.add_argument("--seed", type=int, default=1337)
    main(p.parse_args())
