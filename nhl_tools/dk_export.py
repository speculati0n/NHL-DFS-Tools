"""DraftKings NHL export utilities."""
from __future__ import annotations

import csv
import logging
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

SLOTS: list[str] = ["C1", "C2", "W1", "W2", "W3", "D1", "D2", "G", "UTIL"]
_SLOT_PRIMARY_POS: dict[str, Optional[str]] = {
    "C1": "C",
    "C2": "C",
    "W1": "W",
    "W2": "W",
    "W3": "W",
    "D1": "D",
    "D2": "D",
    "G": "G",
    "UTIL": None,
}
_POS_ALIASES = {
    "LW": "W",
    "RW": "W",
    "LD": "D",
    "RD": "D",
}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlayerRecord:
    name: str
    norm_name: str
    team: Optional[str]
    pos_raw: Optional[str]
    pos_tokens: frozenset[str]
    dk_id: str


@dataclass
class MatchResult:
    display_name: str
    dk_id: str
    record: Optional[PlayerRecord]
    keys_tried: list[tuple[str, Optional[str], Optional[str]]]
    candidates: list[PlayerRecord]
    ambiguous: bool = False


class PlayerIdIndex:
    """Lookup helper for DraftKings player IDs."""

    def __init__(self, records: Iterable[PlayerRecord]):
        self.records: list[PlayerRecord] = list(records)
        self.by_key: dict[tuple[str, Optional[str], Optional[str]], list[PlayerRecord]] = defaultdict(list)
        for record in self.records:
            team = record.team
            pos_tokens = record.pos_tokens or frozenset()
            team_key = team if team else None
            keys: set[tuple[str, Optional[str], Optional[str]]] = set()
            if pos_tokens:
                for pos in pos_tokens:
                    keys.add((record.norm_name, team_key, pos))
                    keys.add((record.norm_name, None, pos))
            keys.add((record.norm_name, team_key, None))
            keys.add((record.norm_name, None, None))
            for key in keys:
                self.by_key[key].append(record)

    def match(
        self,
        display_name: str,
        slot: str,
        team_hint: Optional[str] = None,
        pos_hint: Optional[str] = None,
    ) -> MatchResult:
        norm_name = normalize_name(display_name)
        slot_upper = slot.upper()
        team = team_hint.upper() if team_hint else None
        pos_norm = pos_hint.upper() if pos_hint else None
        pos_tokens_hint = _split_pos_tokens(pos_norm) if pos_norm else set()
        primary_slot_pos = _SLOT_PRIMARY_POS.get(slot_upper)

        keys_tried: list[tuple[str, Optional[str], Optional[str]]] = []
        candidates: list[PlayerRecord] = []

        key_order: list[tuple[str, Optional[str], Optional[str]]] = []
        if team and pos_tokens_hint:
            for pos in pos_tokens_hint:
                key_order.append((norm_name, team, pos))
        if team:
            key_order.append((norm_name, team, None))
        if pos_tokens_hint:
            for pos in pos_tokens_hint:
                key_order.append((norm_name, None, pos))
        key_order.append((norm_name, None, None))

        seen_ids: set[str] = set()
        for key in key_order:
            keys_tried.append(key)
            matches = self.by_key.get(key, [])
            filtered = [rec for rec in matches if rec.dk_id not in seen_ids]
            if filtered:
                candidates.extend(filtered)
                seen_ids.update(rec.dk_id for rec in filtered)
                break

        if not candidates:
            matches = self.by_key.get((norm_name, None, None), [])
            filtered = [rec for rec in matches if rec.dk_id not in seen_ids]
            if filtered:
                candidates.extend(filtered)
                seen_ids.update(rec.dk_id for rec in filtered)

        if not candidates:
            return MatchResult(display_name=display_name, dk_id="", record=None, keys_tried=keys_tried, candidates=[])

        def score_record(record: PlayerRecord) -> tuple[int, str, str, str]:
            score = 0
            if team and record.team == team:
                score += 4
            if pos_tokens_hint and (record.pos_tokens & pos_tokens_hint):
                score += 3
            if primary_slot_pos and primary_slot_pos in record.pos_tokens:
                score += 2
            pos_signature = "/".join(sorted(record.pos_tokens)) if record.pos_tokens else ""
            team_signature = record.team or ""
            return -score, team_signature, pos_signature, record.dk_id

        sorted_candidates = sorted(candidates, key=score_record)
        best = sorted_candidates[0]
        best_score = score_record(best)[0]
        ambiguous = any(score_record(rec)[0] == best_score and rec.dk_id != best.dk_id for rec in sorted_candidates[1:])

        if ambiguous:
            _logger.warning(
                "Ambiguous match for %s (%s): chose %s over %d alternatives",
                display_name,
                slot,
                best.name,
                len(sorted_candidates) - 1,
            )

        return MatchResult(
            display_name=display_name,
            dk_id=best.dk_id,
            record=best,
            keys_tried=keys_tried,
            candidates=sorted_candidates,
            ambiguous=ambiguous,
        )


_PLAYER_INDEX: Optional[PlayerIdIndex] = None
_LAST_MATCH: Optional[MatchResult] = None


def _ensure_index() -> PlayerIdIndex:
    if _PLAYER_INDEX is None:
        raise RuntimeError("Player ID index not loaded. Call load_player_ids() first.")
    return _PLAYER_INDEX


def clean_display_name(name: str | float | None) -> str:
    if name is None:
        return ""
    if isinstance(name, float) and pd.isna(name):
        return ""
    text = str(name)
    text = text.replace("\ufeff", " ").replace("\u200b", " ")
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.strip()
    if text.lower() in {"nan", "none", ""}:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_name(name: str) -> str:
    if not name:
        return ""
    text = clean_display_name(name)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in _SUFFIXES]
    return " ".join(tokens)


def _split_pos_tokens(pos: Optional[str]) -> set[str]:
    if not pos:
        return set()
    tokens: set[str] = set()
    for token in re.split(r"[\s/,-]+", pos):
        token = token.strip().upper()
        if not token:
            continue
        token = _POS_ALIASES.get(token, token)
        tokens.add(token)
    return tokens


def load_player_ids(path: str | os.PathLike[str]) -> PlayerIdIndex:
    global _PLAYER_INDEX
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    lower_cols = {col.lower(): col for col in df.columns}

    id_col = None
    for candidate in ("player_id", "id", "dk_id"):
        if candidate in lower_cols:
            id_col = lower_cols[candidate]
            break
    if id_col is None:
        raise ValueError("Player ID column not found in player IDs file.")

    name_col = None
    for candidate in ("name", "player_name", "fullname", "full_name"):
        if candidate in lower_cols:
            name_col = lower_cols[candidate]
            break
    if name_col is None:
        raise ValueError("Player name column not found in player IDs file.")

    team_col = None
    for candidate in ("team", "teamabbrev", "team_abbrev"):
        if candidate in lower_cols:
            team_col = lower_cols[candidate]
            break

    pos_col = None
    for candidate in ("pos", "position", "positions"):
        if candidate in lower_cols:
            pos_col = lower_cols[candidate]
            break

    records: list[PlayerRecord] = []
    for _, row in df.iterrows():
        if pd.isna(row[name_col]):
            continue
        name = clean_display_name(row[name_col])
        if not name:
            continue
        dk_raw = row[id_col]
        if pd.isna(dk_raw):
            continue
        dk_id = str(dk_raw).strip()
        if not dk_id or dk_id.lower() == "nan":
            continue
        team: Optional[str] = None
        if team_col and not pd.isna(row[team_col]):
            team = str(row[team_col]).strip().upper() or None
        team = team or None
        pos_raw: Optional[str] = None
        if pos_col and not pd.isna(row[pos_col]):
            pos_candidate = str(row[pos_col]).strip().upper()
            pos_raw = pos_candidate or None
        pos_raw = pos_raw or None
        tokens = frozenset(_split_pos_tokens(pos_raw)) if pos_raw else frozenset()
        record = PlayerRecord(
            name=name,
            norm_name=normalize_name(name),
            team=team,
            pos_raw=pos_raw,
            pos_tokens=tokens,
            dk_id=dk_id,
        )
        records.append(record)

    index = PlayerIdIndex(records)
    _PLAYER_INDEX = index
    return index


def last_match_details() -> Optional[MatchResult]:
    return _LAST_MATCH


def map_name_to_id(
    name: str,
    slot: str,
    team_hint: Optional[str] = None,
    pos_hint: Optional[str] = None,
) -> tuple[str, str]:
    global _LAST_MATCH
    index = _ensure_index()
    display_name = clean_display_name(name)
    result = index.match(display_name, slot, team_hint=team_hint, pos_hint=pos_hint)
    _LAST_MATCH = result
    return result.display_name, result.dk_id


def _looks_like_team(token: str) -> bool:
    return token.isalpha() and token.upper() == token and 2 <= len(token) <= 3


def _looks_like_pos(token: str) -> bool:
    token_up = token.upper()
    if token_up in {"C", "W", "D", "G", "UTIL"}:
        return True
    return token_up in _POS_ALIASES


def parse_lineups_any(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse optimizer output that may be in wide or tall format."""

    def _normalize_slot_name(col: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", col.upper())

    def _slot_prefix(norm: str) -> Optional[str]:
        if not norm:
            return None
        if "UTIL" in norm:
            return "UTIL"
        if norm.startswith("G"):
            return "G"
        if norm.startswith("D"):
            return "D"
        if norm.startswith("W") or "W" in norm:
            return "W"
        if norm.startswith("C") or "C" in norm:
            return "C"
        return None

    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Lineup file is empty: {path}") from exc

    column_map: dict[str, str] = {}
    normalized_cols = {col: _normalize_slot_name(col) for col in df.columns}
    prefix_buckets: dict[str, list[str]] = defaultdict(list)
    for col, norm in normalized_cols.items():
        prefix = _slot_prefix(norm)
        if prefix:
            prefix_buckets.setdefault(prefix, []).append(col)

    used_cols: set[str] = set()
    for slot in SLOTS:
        slot_norm = _normalize_slot_name(slot)
        direct_matches = [
            col
            for col, norm in normalized_cols.items()
            if norm == slot_norm and col not in used_cols
        ]
        if direct_matches:
            chosen = direct_matches[0]
            column_map[slot] = chosen
            used_cols.add(chosen)
            continue
        prefix = _slot_prefix(slot_norm)
        if prefix and prefix in prefix_buckets:
            for col in prefix_buckets[prefix]:
                if col not in used_cols:
                    column_map[slot] = col
                    used_cols.add(col)
                    break
        if slot not in column_map and slot in df.columns and slot not in used_cols:
            column_map[slot] = slot
            used_cols.add(slot)

    if len(column_map) == len(SLOTS):
        wide_df = df[[column_map[slot] for slot in SLOTS]].copy()
        wide_df.columns = SLOTS
        wide_df.attrs["slot_hints"] = [{} for _ in range(len(wide_df))]
        return wide_df

    # Tall/stacked parsing.
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    # Detect header row that includes slot/name columns and drop it.
    start_idx = 0
    if rows:
        header = [cell.strip().lower() for cell in rows[0]]
        if "slot" in header and any(col in header for col in ("name", "player")):
            start_idx = 1

    flat_cells: list[str] = []
    for row in rows[start_idx:]:
        for cell in row:
            cell = cell.strip()
            if cell:
                flat_cells.append(cell)

    if not flat_cells:
        raise ValueError("No lineup data detected in file.")

    slot_pattern = re.compile(r"^(?P<slot>C1|C2|W1|W2|W3|D1|D2|G|UTIL)\s*[:\-]?\s*(?P<name>.*)$", re.IGNORECASE)

    entries: list[dict[str, Optional[str]]] = []
    idx = 0
    while idx < len(flat_cells):
        cell = flat_cells[idx]
        match = slot_pattern.match(cell)
        if not match:
            idx += 1
            continue
        slot = match.group("slot").upper()
        leftover = match.group("name").strip()
        idx += 1
        name_tokens: list[str] = []
        team_hint = None
        pos_hint = None
        if leftover:
            name_tokens.append(leftover)
        while idx < len(flat_cells):
            peek = flat_cells[idx]
            if slot_pattern.match(peek):
                break
            peek_upper = peek.upper()
            if team_hint is None and _looks_like_team(peek_upper):
                team_hint = peek_upper
                idx += 1
                continue
            if pos_hint is None and _looks_like_pos(peek_upper):
                pos_hint = peek_upper
                idx += 1
                continue
            name_tokens.append(peek)
            idx += 1
        name = " ".join(name_tokens).strip()
        entries.append({
            "slot": slot,
            "name": name,
            "team_hint": team_hint,
            "pos_hint": pos_hint,
        })

    if len(entries) % len(SLOTS) != 0:
        snippet = " | ".join(flat_cells[:10])
        raise ValueError(
            "Could not reconstruct lineups from stacked file; expected multiples of 9 entries. "
            f"Parsed {len(entries)} entries. Sample: {snippet}"
        )

    lineups: list[dict[str, str]] = []
    hints_per_lineup: list[dict[str, tuple[Optional[str], Optional[str]]]] = []
    for start in range(0, len(entries), len(SLOTS)):
        group = entries[start:start + len(SLOTS)]
        lineup: dict[str, str] = {}
        hint_map: dict[str, tuple[Optional[str], Optional[str]]] = {}
        for entry in group:
            slot = entry["slot"]
            name = entry["name"] or ""
            lineup[slot] = name
            hint_map[slot] = (entry.get("team_hint"), entry.get("pos_hint"))
        missing_slots = [slot for slot in SLOTS if slot not in lineup]
        if missing_slots:
            raise ValueError(f"Lineup missing slots {missing_slots}; cannot construct wide format.")
        ordered_lineup = {slot: lineup.get(slot, "") for slot in SLOTS}
        lineups.append(ordered_lineup)
        hints_per_lineup.append(hint_map)

    wide_df = pd.DataFrame(lineups, columns=SLOTS)
    wide_df.attrs["slot_hints"] = hints_per_lineup
    return wide_df


def format_for_dk(row: dict[str, dict[str, str]]) -> dict[str, str]:
    formatted: dict[str, str] = {}
    for slot in SLOTS:
        data = row.get(slot, {"name": "", "dk_id": ""})
        name = data.get("name", "")
        dk_id = data.get("dk_id", "")
        formatted[slot] = f"{name} ({dk_id})"
    return formatted


def export(
    lineups_path: str | os.PathLike[str],
    ids_path: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
    *,
    strict: bool = False,
    league: str = "NHL",
) -> pd.DataFrame:
    if league.upper() != "NHL":
        raise ValueError("This exporter currently supports only the NHL league.")

    index = load_player_ids(ids_path)
    lineups_df = parse_lineups_any(lineups_path)
    hints = lineups_df.attrs.get("slot_hints") or [{} for _ in range(len(lineups_df))]

    mapped_rows: list[dict[str, dict[str, str]]] = []
    unmatched: list[dict[str, object]] = []

    for idx, row in lineups_df.iterrows():
        hint_map = hints[idx] if idx < len(hints) else {}
        mapped_lineup: dict[str, dict[str, str]] = {}
        for slot in SLOTS:
            raw_name = row.get(slot, "")
            display_name = clean_display_name(raw_name)
            if not display_name:
                mapped_lineup[slot] = {"name": "", "dk_id": ""}
                continue
            team_hint: Optional[str] = None
            pos_hint: Optional[str] = None
            if isinstance(hint_map, dict) and slot in hint_map:
                team_hint, pos_hint = hint_map.get(slot, (None, None))
            slot_pos = _SLOT_PRIMARY_POS.get(slot)
            if not pos_hint and slot_pos:
                pos_hint = slot_pos
            name_out, dk_id = map_name_to_id(display_name, slot, team_hint=team_hint, pos_hint=pos_hint)
            result = last_match_details()
            mapped_lineup[slot] = {"name": name_out, "dk_id": dk_id}
            if result and not dk_id:
                keys_repr = [
                    "/".join(str(part) for part in key if part)
                    for key in result.keys_tried
                ]
                unmatched.append({
                    "slot": slot,
                    "name": name_out,
                    "team_hint": team_hint,
                    "pos_hint": pos_hint,
                    "lineup_index": idx,
                    "keys_tried": " | ".join(keys_repr),
                })
                _logger.warning("Unmatched player: %s (%s)", name_out, slot)
        mapped_rows.append(mapped_lineup)

    formatted_rows = [format_for_dk(row) for row in mapped_rows]
    out_df = pd.DataFrame(formatted_rows, columns=SLOTS)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    if unmatched:
        unmatched_path = out_path.with_name("unmatched_players.csv")
        unmatched_df = pd.DataFrame(unmatched)
        unmatched_df.to_csv(unmatched_path, index=False)
        if strict:
            names = ", ".join(f"{row['name']} ({row['slot']})" for row in unmatched)
            raise RuntimeError(f"Unmatched players: {names}")
    elif strict:
        # Even in strict mode we still succeed if everything matched.
        pass

    return out_df
