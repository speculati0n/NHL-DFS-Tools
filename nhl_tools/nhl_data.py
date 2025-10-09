# nhl_tools/nhl_data.py
import os, re, glob, csv
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick(df: pd.DataFrame, *cands) -> str | None:
    """Robust column picker (FantasyLabs drifts headers a bit)."""
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
    # try to infer F/D (skater type) when present
    typ = None
    if "F" in t:
        typ = "F"
    elif "D" in t:
        typ = "D"
    return (typ, line_no)

def _parse_pp(s):
    """Parse PP unit like 'PP1', '1', 'PP 2'."""
    if s is None or pd.isna(s):
        return None
    t = str(s).upper()
    m = re.search(r"PP\s*([12])", t)
    if m:
        return f"PP{m.group(1)}"
    m = re.search(r"\b([12])\b", t)
    if m:
        return f"PP{m.group(1)}"
    return None

def _read_csv_strict(path: str) -> pd.DataFrame:
    """
    Fallback reader for badly formatted CSVs (e.g., goalie files with extra commas).
    We split each data line at most len(header)-1 times so any extra commas remain
    in the last field, keeping column alignment.
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
    Read a Labs CSV with a robust fallback. We first try pandas normally.
    If the data looks misaligned (e.g., Player is numeric or Salary mostly NaN),
    we re-read via the strict splitter.
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        # catastrophically bad → strict
        return _read_csv_strict(path)

    # Heuristics to detect misalignment
    looks_bad = False

    if "Player" in df.columns:
        # If Player is mostly numeric, it's misparsed.
        s = df["Player"].dropna().head(5)
        if not s.empty and all(pd.api.types.is_numeric_dtype(type(x)) or isinstance(x, (int, float)) for x in s):
            looks_bad = True

    if "Salary" in df.columns:
        # If Salary is mostly NaN, it's suspicious (esp. goalie files).
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
        out["EV_Type"], out["EV_Line"] = zip(*out["FullRaw"].map(_parse_full)) if "FullRaw" in out else (None, None)
        out["PP_Unit"] = out["PP"].map(_parse_pp) if "PP" in out else None

        # canonical single-pos from FantasyLabs file name
        base = os.path.basename(fp).upper()
        if "_NHL_C_" in base: out["PosCanon"] = "C"
        elif "_NHL_W_" in base: out["PosCanon"] = "W"
        elif "_NHL_D_" in base: out["PosCanon"] = "D"
        elif "_NHL_G_" in base: out["PosCanon"] = "G"
        else:
            # fallback: try to read from Pos column
            if c_pos:
                out["PosCanon"] = df[c_pos].astype(str).str.extract(r"(C|W|D|G)")[0]
            else:
                out["PosCanon"] = None

        parts.append(out)

    # Concatenate and clean
    full = pd.concat(parts, ignore_index=True)

    # Basic sanity drops
    full = full.dropna(subset=["Team", "Salary", "Proj", "PosCanon"])
    full = full.drop_duplicates(subset=["Name", "Team", "PosCanon"], keep="first").reset_index(drop=True)

    # Game key for bring-backs
    def _gk(row):
        a, b = sorted([row["Team"], row["Opp"]]) if row["Opp"] and row["Opp"] != "None" else (row["Team"], None)
        return f"{a}@{b}" if b else None

    full["GameKey"] = full.apply(_gk, axis=1)

    # sanity flags
    full["IsSkater"] = full["PosCanon"].isin(["C", "W", "D"])
    full["IsGoalie"] = (full["PosCanon"] == "G")
    return full

# DK rules
DK_ROSTER = dict(C=2, W=3, D=2, G=1, UTIL=1)
DK_SALARY_CAP = 50000
