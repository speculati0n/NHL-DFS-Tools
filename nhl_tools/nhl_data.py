# nhl_tools/nhl_data.py
import os, re, glob
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

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
    if s is None: return None
    if pd.isna(s): return None
    if isinstance(s, (int, float)): 
        try:
            return int(s)
        except Exception:
            try:
                return float(s)
            except Exception:
                return None
    # strip $ , spaces etc
    m = re.sub(r"[^\d.\-]", "", str(s))
    if m == "" or m == "-" or m == "--":
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
    # Common patterns: "L1", "L2", "L1D", "L1F", "1", "EV 1"
    m = re.search(r"(?:L)?\s*(\d)", t)
    if not m:
        return (None, None)
    line_no = int(m.group(1))
    # try to infer F/D (skater type) when present (not required for optimizer)
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
    """
    Fallback reader for badly formatted CSVs (e.g., goalie files with extra commas).
    Split each line at most len(header)-1 times so extra commas stay in the last field.
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
    """
    Read a Labs CSV with a robust fallback. First try pandas normally.
    If the data looks misaligned (e.g., Player mostly numeric or Salary mostly NaN),
    re-read via the strict splitter.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return _read_csv_strict(path)

    looks_bad = False
    if "Player" in df.columns:
        s = df["Player"].dropna().head(5)
        if not s.empty and all(str(x).strip().replace(".", "", 1).isdigit() for x in s):
            looks_bad = True
    if "Salary" in df.columns:
        na_ratio = df["Salary"].isna().mean()
        if na_ratio > 0.6:
            looks_bad = True
    if looks_bad:
        df = _read_csv_strict(path)
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_labs_for_date(stats_dir: str, ymd: str) -> pd.DataFrame:
    """
    Read FantasyLabs NHL CSVs for a single date (YYYY-MM-DD).
    Expected files in stats_dir:
      fantasylabs_player_data_NHL_[C|W|D|G]_YYYYMMDD.csv
    """
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
        c_pos    = _pick(df, "Pos", "Position")
        c_team   = _pick(df, "Team", "Tm")
        c_opp    = _pick(df, "Opp", "Opponent")
        c_sal    = _pick(df, "Salary", "Sal")
        c_proj   = _pick(df, "Proj", "Projection", "My Proj", "My")
        c_own    = _pick(df, "Own", "Ownership", "Proj Own", "My Own", "Ownership%", "Own%")
        c_full   = _pick(df, "Full", "Even Strength Line", "EV Line", "Line")
        c_pp     = _pick(df, "PP", "Power Play", "PP Unit")

        if not c_player or not c_team or not c_sal or not c_proj:
            raise ValueError(
                f"Missing required columns in {os.path.basename(fp)} — "
                f"need Player/Team/Salary/Proj-like columns; got: {list(df.columns)[:10]}..."
            )

        out = pd.DataFrame({
            "Name":  df[c_player].astype(str).str.strip(),
            "Team":  df[c_team].astype(str).str.strip(),
            "Opp":   df[c_opp].astype(str).str.strip() if c_opp else None,
            "Salary": df[c_sal].map(_coerce_num),
            "Proj":   df[c_proj].map(_coerce_num),
            "Own":    df[c_own].map(_coerce_num) if c_own else np.nan,
            "FullRaw": df[c_full] if c_full else None,
            "PP":     df[c_pp] if c_pp else None,
        })

        # derived EV / PP features
        if "FullRaw" in out:
            ev_type, ev_line = zip(*out["FullRaw"].map(_parse_full))
            out["EV_Type"] = list(ev_type)
            out["EV_Line"] = list(ev_line)
        else:
            out["EV_Type"], out["EV_Line"] = (None, None)

        out["PP_Unit"] = out["PP"].map(_pp_to_int) if "PP" in out else None  # ← normalize to ints 1/2

        # canonical single-pos from FantasyLabs file name
        base = os.path.basename(fp).upper()
        if "_NHL_C_" in base: out["PosCanon"] = "C"
        elif "_NHL_W_" in base: out["PosCanon"] = "W"
        elif "_NHL_D_" in base: out["PosCanon"] = "D"
        elif "_NHL_G_" in base: out["PosCanon"] = "G"
        else:
            if c_pos:
                out["PosCanon"] = df[c_pos].astype(str).str.extract(r"(C|W|D|G)")[0]
            else:
                out["PosCanon"] = None

        # Ensure all expected columns exist (avoids concat all-NA warnings in newer pandas)
        for col in ["Opp", "Own", "FullRaw", "EV_Type", "EV_Line", "PP", "PP_Unit", "PosCanon"]:
            if col not in out.columns:
                out[col] = np.nan

        parts.append(out)

    # Concatenate and clean
    full = pd.concat(parts, ignore_index=True)

    # Basic sanity drops
    full = full.dropna(subset=["Team", "Salary", "Proj", "PosCanon"])
    full = full.drop_duplicates(subset=["Name", "Team", "PosCanon"], keep="first").reset_index(drop=True)

    # Game key for bring
