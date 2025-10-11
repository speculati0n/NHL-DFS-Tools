import logging
import os
import re
import glob
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _pick(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lowered:
            return lowered[key]
    return None


def _norm_date_str(x: str) -> str:
    s = str(x).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    return s[:10]


def _coerce_num(s):
    if s is None:
        return None
    if pd.isna(s):
        return None
    if isinstance(s, (int, float)):
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return None
    m = re.sub(r"[^\d.\-]", "", str(s))  # strip $, commas, spaces etc.
    if m in ("", "-", "--"):
        return None
    try:
        if "." in m:
            v = float(m)
            return int(v) if v.is_integer() else v
        return int(m)
    except Exception:
        return None


def _parse_full(s):
    """Parse FantasyLabs Full/Line column (EV lines) into normalized token."""
    if s is None or pd.isna(s):
        return None
    t = str(s).strip().upper()
    t = t.replace("EV", "").replace("LINE", "").strip()
    m = re.search(r"(\d)\s*([FD])?", t)
    if not m:
        m = re.search(r"([FD])\s*(\d)", t)
        if not m:
            return None
        parts = m.groups()
        if len(parts) == 2:
            typ = parts[0]
            line_no = parts[1]
        else:
            typ = None
            line_no = parts[0]
    else:
        line_no = m.group(1)
        typ = m.group(2)
    try:
        line_no = int(line_no)
    except Exception:
        return None
    typ = (typ or ("F" if "F" in t else ("D" if "D" in t else ""))).upper()
    if typ not in {"F", "D", ""}:
        typ = ""
    return f"{line_no}{typ}" if typ else f"{line_no}"


def _pp_to_int(s) -> Optional[int]:
    """Normalize PP unit to integer (1/2) from 'PP1', '1', 'PP 2', etc."""
    if s is None or pd.isna(s):
        return None
    t = str(s).strip().upper()
    m = re.search(r"(\d+)", t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _read_csv_strict(path: str) -> pd.DataFrame:
    """Fallback reader for badly formatted CSVs.

    Some FantasyLabs NHL exports occasionally include unescaped commas in the goalie
    file.  The pandas default CSV parser will misalign the columns in that case.  To
    keep the import resilient we perform a capped split that keeps any extra commas
    inside the final column instead of creating new ones.
    """

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    if not lines:
        return pd.DataFrame()

    header = lines[0].split(",")
    n = len(header)
    rows = []
    for ln in lines[1:]:
        parts = ln.split(",", n - 1)
        if len(parts) < n:
            parts += [""] * (n - len(parts))
        rows.append(parts[:n])
    return pd.DataFrame(rows, columns=header)


def _read_labs_csv(path: str) -> pd.DataFrame:
    """Read a Labs CSV with a robust fallback."""

    try:
        df = pd.read_csv(path)
    except Exception:
        return _read_csv_strict(path)

    looks_bad = False
    if "Player" in df.columns:
        s = df["Player"].dropna().head(5)
        if not s.empty and all(str(x).strip().replace(".", "", 1).isdigit() for x in s):
            looks_bad = True
    if "Salary" in df.columns and df["Salary"].isna().mean() > 0.6:
        looks_bad = True
    if looks_bad:
        df = _read_csv_strict(path)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def load_labs_for_date(stats_dir: str, ymd: str) -> pd.DataFrame:
    """Read FantasyLabs NHL CSVs for a single date (YYYY-MM-DD)."""

    ymd_compact = ymd.replace("-", "")
    fps = []
    for pos in ["C", "W", "D", "G"]:
        pat = os.path.join(stats_dir, f"fantasylabs_player_data_NHL_{pos}_{ymd_compact}.csv")
        matches = glob.glob(pat)
        if not matches:
            raise FileNotFoundError(f"Missing Labs file for position {pos}: {pat}")
        fps.extend(matches)

    parts = []
    for fp in fps:
        df = _read_labs_csv(fp)

        c_player = _pick(df, "Player", "Name")
        c_pos = _pick(df, "Pos", "Position")
        c_team = _pick(df, "Team", "Tm")
        c_opp = _pick(df, "Opp", "Opponent")
        c_sal = _pick(df, "Salary", "Sal")
        c_proj = _pick(df, "Proj", "Projection", "My Proj", "My")
        c_ceil = _pick(df, "Ceil", "Ceiling", "Ceil.Projection", "Ceiling Proj")
        c_own = _pick(df, "Own", "Ownership", "Proj Own", "My Own", "Ownership%", "Own%")
        c_full = _pick(df, "Full", "Even Strength Line", "EV Line", "Line")
        c_pp = _pick(df, "PP", "Power Play", "PP Unit")

        if not c_player or not c_team or not c_sal or not c_proj:
            raise ValueError(
                "Missing required columns in "
                f"{os.path.basename(fp)} — need Player/Team/Salary/Proj-like columns; "
                f"got: {list(df.columns)[:10]}..."
            )

        out = pd.DataFrame(
            {
                "Name": df[c_player].astype(str).str.strip(),
                "Team": df[c_team].astype(str).str.strip(),
                "Opp": df[c_opp].astype(str).str.strip() if c_opp else np.nan,
                "Salary": df[c_sal].map(_coerce_num),
                "Proj": df[c_proj].map(_coerce_num),
                "Own": df[c_own].map(_coerce_num) if c_own else np.nan,
                "Ceiling": df[c_ceil].map(_coerce_num) if c_ceil else np.nan,
                "Full": df[c_full] if c_full else np.nan,
                "PP": df[c_pp] if c_pp else np.nan,
            }
        )

        # derived EV / PP features
        if "Full" in out:
            out["FullNorm"] = out["Full"].map(_parse_full)
        else:
            out["FullNorm"] = np.nan
        out["PP_Unit"] = out["PP"].map(_pp_to_int)

        # canonical single-pos from FantasyLabs file name
        base = os.path.basename(fp).upper()
        if "_NHL_C_" in base:
            out["PosCanon"] = "C"
        elif "_NHL_W_" in base:
            out["PosCanon"] = "W"
        elif "_NHL_D_" in base:
            out["PosCanon"] = "D"
        elif "_NHL_G_" in base:
            out["PosCanon"] = "G"
        else:
            if c_pos:
                out["PosCanon"] = df[c_pos].astype(str).str.extract(r"(C|W|D|G)")[0]
            else:
                out["PosCanon"] = np.nan

        # Ensure expected columns exist (avoid concat-NA warning in new pandas)
        for col in ["Opp", "Own", "Ceiling", "Full", "FullNorm", "PP", "PP_Unit", "PosCanon"]:
            if col not in out.columns:
                out[col] = np.nan

        parts.append(out)

    full = pd.concat(parts, ignore_index=True)

    # Basic sanity
    full = full.dropna(subset=["Team", "Salary", "Proj", "PosCanon"])
    full = full.drop_duplicates(subset=["Name", "Team", "PosCanon"], keep="first").reset_index(drop=True)

    # Game key for bring-backs
    def _gk(row):
        if pd.notna(row["Opp"]) and row["Opp"] != "None":
            a, b = sorted([str(row["Team"]), str(row["Opp"])])
            return f"{a}@{b}"
        return None

    full["GameKey"] = full.apply(_gk, axis=1)

    # flags
    full["IsSkater"] = full["PosCanon"].isin(["C", "W", "D"])
    full["IsGoalie"] = full["PosCanon"] == "G"
    full["FullNorm"] = full["FullNorm"].replace({"": np.nan})

    return full


# DraftKings rules (shared by optimizer and simulator)
DK_ROSTER = dict(C=2, W=3, D=2, G=1, UTIL=1)
DK_SALARY_CAP = 50000


# ──────────────────────────────────────────────────────────────────────────────
# Ownership + grouping helpers
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name or ""))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def attach_ownership(df: pd.DataFrame, ownership_cfg: Optional[Dict]) -> pd.DataFrame:
    if not ownership_cfg:
        df["Ownership"] = np.nan
        return df

    enabled = bool(ownership_cfg.get("enable"))
    path = str(ownership_cfg.get("file") or "")
    column = ownership_cfg.get("column", "proj_own")
    if not enabled or not path or not os.path.exists(path):
        df["Ownership"] = np.nan
        return df

    own_df = pd.read_csv(path)
    col_name = _pick(own_df, column, column.replace("_", " "))
    if not col_name:
        raise ValueError(f"Ownership column '{column}' not found in {path}.")

    team_col = _pick(own_df, "team", "Team", "teamabbr", "team_abbrev", "TeamAbbrev")
    pos_col = _pick(own_df, "pos", "position", "Pos", "Position")
    name_col = _pick(own_df, "name", "player", "Player", "Name")
    if not name_col:
        raise ValueError("Ownership file missing a recognizable name column.")

    own_df = own_df[[name_col] + ([team_col] if team_col else []) + ([pos_col] if pos_col else []) + [col_name]].copy()
    own_df.columns = ["Name", "Team", "Pos", "Ownership"][: len(own_df.columns)]
    own_df["NameNorm"] = own_df["Name"].map(_normalize_name)
    if "Team" in own_df.columns:
        own_df["Team"] = own_df["Team"].astype(str).str.upper().str.strip()
    if "Pos" in own_df.columns:
        own_df["Pos"] = own_df["Pos"].astype(str).str.upper().str.strip()

    df = df.copy()
    df["NameNorm"] = df["Name"].map(_normalize_name)
    df["TeamNorm"] = df["Team"].astype(str).str.upper().str.strip()
    df["PosNorm"] = df["PosCanon"].astype(str).str.upper().str.strip()

    merged = pd.merge(
        df,
        own_df,
        left_on=["NameNorm", "TeamNorm", "PosNorm"],
        right_on=["NameNorm", "Team", "Pos"],
        how="left",
        suffixes=("", "_own"),
    )
    merged["Ownership"] = pd.to_numeric(merged["Ownership"], errors="coerce")
    merged.drop(columns=[c for c in ["Team", "Pos"] if c in merged.columns], inplace=True)
    merged.drop(columns=["NameNorm", "TeamNorm", "PosNorm"], inplace=True)
    return merged


def load_player_pool(stats_dir: str, ymd: str, ownership_cfg: Optional[Dict] = None) -> pd.DataFrame:
    base = load_labs_for_date(stats_dir, ymd)
    base = base.rename(columns={"Name": "Name", "Team": "Team"})
    base["Full"] = base["FullNorm"]
    base.drop(columns=[c for c in ["FullNorm"] if c in base.columns], inplace=True)

    base["full_line_group"] = np.where(
        base["IsSkater"],
        base.apply(
            lambda r: f"{r['Team']}:{r['Full']}" if pd.notna(r.get("Full")) else np.nan,
            axis=1,
        ),
        np.nan,
    )
    base["pp_unit"] = np.where(base["IsSkater"], base["PP_Unit"], np.nan)

    base = attach_ownership(base, ownership_cfg)

    base.rename(
        columns={
            "PosCanon": "Pos",
            "Proj": "Proj",
        },
        inplace=True,
    )

    ordered_cols = [
        "Name",
        "Pos",
        "Team",
        "Opp",
        "Salary",
        "Proj",
        "Ceiling",
        "Full",
        "PP",
        "full_line_group",
        "pp_unit",
        "Ownership",
        "IsSkater",
        "IsGoalie",
        "GameKey",
    ]
    for c in ordered_cols:
        if c not in base.columns:
            base[c] = np.nan
    base = base[ordered_cols + [c for c in base.columns if c not in ordered_cols]].copy()
    return base


def group_evline(players: pd.DataFrame) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = {}
    skaters = players[players["IsSkater"] & players["full_line_group"].notna()]
    for idx, row in skaters.iterrows():
        key = (row["Team"], row["Full"])
        groups.setdefault(key, []).append(idx)
    return groups


def group_pp1(players: pd.DataFrame) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    pp1 = players[players["IsSkater"] & (players["pp_unit"] == 1)]
    for idx, row in pp1.iterrows():
        groups.setdefault(row["Team"], []).append(idx)
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Simulator helpers (public)
# ──────────────────────────────────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Public wrapper around the name normalizer."""

    return _normalize_name(name)


def normalize_position(pos: Optional[str]) -> str:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return ""
    text = str(pos).strip().upper()
    matches = re.findall(r"(C|W|D|G)", text)
    if matches:
        return matches[0]
    if text in {"LW", "RW"}:
        return "W"
    return text


def _empty_player_reference() -> pd.DataFrame:
    cols = [
        "Name",
        "Pos",
        "Team",
        "Opp",
        "Salary",
        "Proj",
        "Ceiling",
        "Full",
        "PP",
        "Ownership",
        "Actual",
        "PlayerID",
        "full_line_group",
        "pp_unit",
    ]
    return pd.DataFrame(columns=cols)


def load_player_reference(path: str) -> pd.DataFrame:
    """Load a generic player pool CSV and normalize common columns."""

    if not path or not os.path.exists(path):
        LOG.warning("Player pool file %s missing; returning empty frame", path)
        return _empty_player_reference()

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        frames = []
        for fp in files:
            try:
                frames.append(pd.read_csv(fp))
            except Exception as exc:
                LOG.warning("Failed to read %s: %s", fp, exc)
        if not frames:
            LOG.warning("No readable CSV files found under %s", path)
            return _empty_player_reference()
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)
    if df.empty:
        return _empty_player_reference()

    name_col = _pick(df, "Name", "Player", "Player Name")
    pos_col = _pick(df, "Pos", "Position")
    team_col = _pick(df, "Team", "TeamAbbrev", "Team Abbrev", "Tm")
    opp_col = _pick(df, "Opp", "Opponent", "Opp Team")
    salary_col = _pick(df, "Salary", "Sal")
    proj_col = _pick(df, "Proj", "Projection", "Fpts Proj", "Fpts")
    ceil_col = _pick(df, "Ceiling", "Ceil")
    own_col = _pick(df, "Ownership", "Own", "Proj Own", "Proj. Own", "Own%")
    full_col = _pick(df, "Full", "EV", "Even Strength Line", "Line")
    pp_col = _pick(df, "PP", "Power Play", "PP Unit")
    actual_col = _pick(df, "Actual", "Fpts Act", "Fpts", "Points", "Score")
    id_col = _pick(df, "PlayerID", "ID", "Player Id", "player_id")

    out = pd.DataFrame(
        {
            "Name": df[name_col].astype(str).str.strip() if name_col else df.index.astype(str),
            "Pos": df[pos_col] if pos_col else np.nan,
            "Team": df[team_col] if team_col else np.nan,
            "Opp": df[opp_col] if opp_col else np.nan,
            "Salary": df[salary_col].map(_coerce_num) if salary_col else np.nan,
            "Proj": df[proj_col].map(_coerce_num) if proj_col else np.nan,
            "Ceiling": df[ceil_col].map(_coerce_num) if ceil_col else np.nan,
            "Full": df[full_col] if full_col else np.nan,
            "PP": df[pp_col] if pp_col else np.nan,
            "Ownership": df[own_col].map(_coerce_num) if own_col else np.nan,
            "Actual": df[actual_col].map(_coerce_num) if actual_col else np.nan,
            "PlayerID": df[id_col] if id_col else np.nan,
        }
    )

    out["Pos"] = out["Pos"].map(normalize_position)
    out["Team"] = out["Team"].astype(str).str.upper().str.strip()
    out["Opp"] = out["Opp"].astype(str).str.upper().str.strip()
    out.loc[out["Opp"].isin(["", "NONE", "NAN"]), "Opp"] = np.nan

    out["Full"] = out["Full"].astype(str).replace({"nan": np.nan, "None": np.nan})
    out["Full"] = out["Full"].where(out["Full"].notna(), None)
    out["Full"] = out["Full"].map(lambda x: _parse_full(x) if pd.notna(x) else np.nan)
    out["pp_unit"] = out["PP"].map(_pp_to_int)

    out["Name"] = out["Name"].astype(str).str.strip()
    out["NameNorm"] = out["Name"].map(_normalize_name)

    def _game_key(row):
        team = row["Team"]
        opp = row["Opp"]
        if pd.notna(team) and pd.notna(opp) and opp:
            a, b = sorted([str(team), str(opp)])
            return f"{a}@{b}"
        return None

    out["GameKey"] = out.apply(_game_key, axis=1)
    out["IsSkater"] = out["Pos"].isin(["C", "W", "D"])
    out["IsGoalie"] = out["Pos"] == "G"
    out["full_line_group"] = np.where(
        out["IsSkater"],
        out.apply(
            lambda r: f"{r['Team']}:{r['Full']}" if pd.notna(r.get("Full")) else np.nan,
            axis=1,
        ),
        np.nan,
    )
    return out


def load_player_reference_for_date(path: str, ymd: Optional[str]) -> pd.DataFrame:
    df = load_player_reference(path)
    if ymd:
        date_col = _pick(df, "date", "Date", "ymd", "YMD")
        if date_col:
            df = df.copy()
            df["_ymd_norm"] = df[date_col].map(_norm_date_str)
            df = df[df["_ymd_norm"] == ymd].drop(columns=["_ymd_norm"])
    nm = _pick(df, "Name", "Player")
    tm = _pick(df, "Team", "TeamAbbrev", "team")
    pm = _pick(df, "Pos", "Position", "position")
    if nm and tm and pm:
        full_col = _pick(df, "Full")
        pp_col = _pick(df, "PP")
        df = df.copy()
        df["has_fullpp"] = df[full_col].notna().astype(int) if full_col else 0
        if pp_col:
            df["has_fullpp"] += df[pp_col].notna().astype(int)
        df = (
            df.sort_values(["has_fullpp", "Proj"], ascending=[False, False])
              .drop_duplicates(subset=[nm, tm, pm], keep="first")
              .drop(columns=["has_fullpp"], errors="ignore")
        )
    return df


def filter_by_slate_date(
    df: pd.DataFrame, date_col_candidates: Tuple[str, ...] = ("date", "Date"), ymd: Optional[str] = None
) -> pd.DataFrame:
    """Filter a player reference frame to a single slate date if available."""

    if df.empty or not ymd:
        return df

    date_col = _pick(df, *date_col_candidates)
    if not date_col:
        LOG.warning("Player reference missing date column; unable to filter for %s", ymd)
        return df

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().all():
        LOG.warning("Unable to parse %s values in %s for date filtering", date_col, ymd)
        return df

    mask = dates.dt.strftime("%Y-%m-%d") == ymd
    if not mask.any():
        LOG.warning("No player reference rows matched slate date %s; leaving unfiltered", ymd)
        return df
    return df.loc[mask].copy()


def _dedupe_player_reference(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    work = df.copy()
    date_col = _pick(work, "date", "Date")
    if date_col:
        work["__sort_date__"] = pd.to_datetime(work[date_col], errors="coerce")
    else:
        work["__sort_date__"] = pd.NaT

    work["__sort_proj__"] = pd.to_numeric(work.get("Proj"), errors="coerce").fillna(-np.inf)
    work["__sort_idx__"] = np.arange(len(work))

    work = work.sort_values(
        by=["__sort_proj__", "__sort_date__", "__sort_idx__"], ascending=[False, False, False]
    )

    deduped = work.drop_duplicates(subset=["Name", "Team", "Pos"], keep="first")
    return deduped.drop(columns=["__sort_proj__", "__sort_date__", "__sort_idx__"], errors="ignore")


def _meta_pick(meta: Dict[str, object], *keys: str):
    for key in keys:
        if key in meta and not pd.isna(meta[key]):
            return meta[key]
    return None


def normalize_lineup_player(
    name: str,
    meta: Dict[str, object],
    player_lookup: Dict[str, List[pd.Series]],
) -> Dict[str, object]:
    norm_name = normalize_name(name)
    meta_lower = {str(k).lower(): v for k, v in meta.items()}

    meta_team = _meta_pick(meta_lower, "team", "teamabbr", "team_abbrev", "tm")
    meta_pos = _meta_pick(meta_lower, "pos", "position", "slot", "roster")
    meta_opp = _meta_pick(meta_lower, "opp", "opponent")
    meta_salary = _coerce_num(_meta_pick(meta_lower, "salary", "sal"))
    meta_proj = _coerce_num(_meta_pick(meta_lower, "proj", "projection", "fpts proj", "fpts"))
    meta_ceil = _coerce_num(_meta_pick(meta_lower, "ceiling", "ceil"))
    meta_actual = _coerce_num(_meta_pick(meta_lower, "actual", "fpts act", "points", "score"))
    meta_own = _coerce_num(_meta_pick(meta_lower, "ownership", "own", "proj own", "own%"))

    candidates = player_lookup.get(norm_name, [])
    team_upper = str(meta_team).upper().strip() if meta_team else ""
    pos_upper = normalize_position(meta_pos)

    def _prefer(pool: List[pd.Series], predicate) -> List[pd.Series]:
        if not pool:
            return pool
        matches = [row for row in pool if predicate(row)]
        return matches if matches else pool

    narrowed = candidates
    if team_upper:
        narrowed = _prefer(
            narrowed,
            lambda row: str(row.get("Team", "")).upper().strip() == team_upper,
        )
    if pos_upper:
        narrowed = _prefer(narrowed, lambda row: normalize_position(row.get("Pos")) == pos_upper)

    def _has_line_info(row: pd.Series) -> bool:
        full_val = row.get("Full")
        has_full = pd.notna(full_val) and str(full_val).strip() not in {"", "nan", "None"}
        return has_full or pd.notna(row.get("pp_unit"))

    narrowed = _prefer(narrowed, _has_line_info)

    pool = narrowed if narrowed else candidates

    best_row: Optional[pd.Series] = None
    best_score = -1
    for row in pool:
        score = 0
        row_team = str(row.get("Team", "")).upper().strip()
        row_pos = normalize_position(row.get("Pos"))
        if team_upper and row_team == team_upper:
            score += 3
        if pos_upper and row_pos == pos_upper:
            score += 2
        salary = row.get("Salary")
        if meta_salary is not None and pd.notna(salary) and abs(float(salary) - meta_salary) <= 100:
            score += 1
        if _has_line_info(row):
            score += 0.5
        if score > best_score:
            best_score = score
            best_row = row

    def _fallback(value, row_value):
        return value if value not in (None, "", np.nan) else row_value

    row = best_row if best_row is not None else (candidates[0] if candidates else pd.Series())
    row_pos = normalize_position(row.get("Pos")) if isinstance(row, pd.Series) else ""
    row_team = str(row.get("Team", "")).upper().strip() if isinstance(row, pd.Series) else ""
    row_opp = str(row.get("Opp", "")).upper().strip() if isinstance(row, pd.Series) else ""

    salary = _fallback(meta_salary, row.get("Salary") if isinstance(row, pd.Series) else None)
    projection = _fallback(meta_proj, row.get("Proj") if isinstance(row, pd.Series) else None)
    ceiling = _fallback(meta_ceil, row.get("Ceiling") if isinstance(row, pd.Series) else None)
    actual = _fallback(meta_actual, row.get("Actual") if isinstance(row, pd.Series) else None)
    ownership = _fallback(meta_own, row.get("Ownership") if isinstance(row, pd.Series) else None)
    full = row.get("Full") if isinstance(row, pd.Series) else None
    pp_unit = row.get("pp_unit") if isinstance(row, pd.Series) else None
    if pd.notna(pp_unit) and str(pp_unit).strip() != "":
        try:
            pp_unit_value = int(float(str(pp_unit).strip()))
        except (TypeError, ValueError):
            pp_unit_value = None
    else:
        pp_unit_value = None
    player_id = row.get("PlayerID") if isinstance(row, pd.Series) else None

    def _clean_float(val):
        if val in (None, "", "nan"):
            return None
        try:
            return float(val)
        except Exception:
            return None

    data = {
        "name": str(name).strip(),
        "position": normalize_position(_fallback(pos_upper, row_pos)) or "",
        "team": (_fallback(team_upper, row_team) or "").upper(),
        "opp": (_fallback(str(meta_opp).upper().strip() if meta_opp else None, row_opp) or None),
        "salary": _clean_float(salary),
        "projection": _clean_float(projection),
        "ceiling": _clean_float(ceiling),
        "actual": _clean_float(actual),
        "ownership": _clean_float(ownership),
        "full": full if full not in (np.nan, "nan") else None,
        "pp_unit": pp_unit_value,
        "player_id": str(player_id) if player_id not in (None, np.nan, "") else None,
    }

    if data["opp"] == "":
        data["opp"] = None

    return data


def apply_external_ownership(
    players_df: pd.DataFrame,
    ownership_path: Optional[str],
    ownership_column: Optional[str] = None,
) -> pd.DataFrame:
    if not ownership_path or not os.path.exists(ownership_path):
        if players_df.empty:
            return pd.DataFrame(columns=["Name", "Team", "Pos", "Ownership"])
        out = players_df.copy()
        out["Ownership"] = np.nan
        return out

    own_df = pd.read_csv(ownership_path)
    if own_df.empty:
        if players_df.empty:
            return pd.DataFrame(columns=["Name", "Team", "Pos", "Ownership"])
        players_df = players_df.copy()
        players_df["Ownership"] = np.nan
        return players_df

    target_col = None
    if ownership_column:
        target_col = _pick(own_df, ownership_column)
    if not target_col:
        target_col = _pick(own_df, "Ownership", "Own", "Proj Own", "Proj. Own", "Own%")
    if not target_col:
        raise ValueError("Unable to locate ownership column in ownership file")

    name_col = _pick(own_df, "Name", "Player", "Player Name")
    if not name_col:
        raise ValueError("Ownership file missing a name column")
    team_col = _pick(own_df, "Team", "TeamAbbrev", "Team Abbrev", "Tm")
    pos_col = _pick(own_df, "Pos", "Position")

    normalized = pd.DataFrame(
        {
            "Name": own_df[name_col].astype(str).str.strip(),
            "Team": own_df[team_col].astype(str).str.upper().str.strip() if team_col else np.nan,
            "Pos": own_df[pos_col].astype(str).str.strip() if pos_col else np.nan,
            "Ownership": pd.to_numeric(own_df[target_col], errors="coerce"),
        }
    )
    normalized["NameNorm"] = normalized["Name"].map(_normalize_name)
    normalized["TeamNorm"] = normalized["Team"].astype(str).str.upper().str.strip()
    normalized["PosNorm"] = normalized["Pos"].map(normalize_position)

    if players_df.empty:
        return normalized[["Name", "Team", "Pos", "Ownership"]]

    merged = players_df.copy()
    existing_ownership = merged["Ownership"] if "Ownership" in merged.columns else None
    if "Ownership" in merged.columns:
        merged = merged.drop(columns=["Ownership"])
    merged["NameNorm"] = merged["Name"].map(_normalize_name)
    merged["TeamNorm"] = merged["Team"].astype(str).str.upper().str.strip()
    merged["PosNorm"] = merged["Pos"].map(normalize_position)

    merge_cols = ["NameNorm"]
    if normalized["TeamNorm"].notna().any():
        merge_cols.append("TeamNorm")
    if normalized["PosNorm"].notna().any():
        merge_cols.append("PosNorm")

    merged = merged.merge(
        normalized[merge_cols + ["Ownership"]],
        on=merge_cols,
        how="left",
    )

    if "Ownership" not in merged.columns:
        merged["Ownership"] = np.nan

    if existing_ownership is not None:
        merged["Ownership"] = merged["Ownership"].fillna(existing_ownership)

    if merged["Ownership"].isna().any():
        fallback = normalized.groupby("NameNorm")["Ownership"].mean()
        missing_mask = merged["Ownership"].isna()
        merged.loc[missing_mask, "Ownership"] = merged.loc[missing_mask, "NameNorm"].map(fallback)
        still_missing = merged[missing_mask & merged["Ownership"].isna()]
        if not still_missing.empty:
            unresolved = sorted(set(still_missing["Name"].tolist()))
            LOG.warning("Ownership unresolved for %d players: %s", len(unresolved), ", ".join(unresolved[:10]))

    merged.drop(columns=["NameNorm", "TeamNorm", "PosNorm"], inplace=True)
    return merged
