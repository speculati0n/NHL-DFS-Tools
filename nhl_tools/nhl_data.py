import os
import re
import glob
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _pick(df: pd.DataFrame, *cands) -> Optional[str]:
    """Robust column picker (FantasyLabs headers drift)."""
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        k = c.lower()
        if k in low:
            return low[k]
    return None


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
    """Parse FantasyLabs Full/Line column (EV lines) into (type, line_no)."""
    if s is None or pd.isna(s):
        return (None, None)
    t = str(s).strip().upper()
    m = re.search(r"(?:L)?\s*(\d)", t)  # L1 / 1 / EV 1 etc.
    if not m:
        return (None, None)
    line_no = int(m.group(1))
    typ = None
    if "F" in t:
        typ = "F"
    elif "D" in t:
        typ = "D"
    return (typ, line_no)


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
                "FullRaw": df[c_full] if c_full else np.nan,
                "PP": df[c_pp] if c_pp else np.nan,
            }
        )

        # derived EV / PP features
        ev_type, ev_line = zip(*out["FullRaw"].map(_parse_full)) if "FullRaw" in out else ([], [])
        out["EV_Type"] = list(ev_type)
        out["EV_Line"] = list(ev_line)
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
        for col in ["Opp", "Own", "FullRaw", "EV_Type", "EV_Line", "PP", "PP_Unit", "PosCanon"]:
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
    return full


# DraftKings rules (shared by optimizer and simulator)
DK_ROSTER = dict(C=2, W=3, D=2, G=1, UTIL=1)
DK_SALARY_CAP = 50000
