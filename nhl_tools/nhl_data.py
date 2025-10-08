import os, re, glob
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Robust column pick (FantasyLabs has light header drift across seasons)
def _pick(df: pd.DataFrame, *cands) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        k = c.lower()
        if k in low: return low[k]
    return None

def _coerce_num(s):
    if s is None: return None
    if pd.isna(s): return None
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return None

def _norm_team(s):
    if not isinstance(s, str): return None
    s = s.strip().upper()
    # strip @ or vs
    s = re.sub(r"[^A-Z]", "", s)
    return s or None

def _opp_to_team(s):
    if not isinstance(s, str): return None
    m = re.search(r"@?([A-Z]{2,3})", s.upper())
    return m.group(1) if m else None

def _parse_full(v:str|None) -> Tuple[str|None,int|None]:
    """Return ('F'|'D', line_num) from FantasyLabs 'Full' like '1F','2F','1D'. None if NA."""
    if not isinstance(v, str): return (None, None)
    v = v.strip().upper()
    m = re.match(r"^([123])([FD])$", v)
    if not m: return (None, None)
    line = int(m.group(1))
    t = m.group(2)
    return (t, line)

def _parse_pp(v) -> int|None:
    if v is None or (isinstance(v,float) and np.isnan(v)): return None
    try:
        return int(str(v).strip())
    except Exception:
        return None

def load_labs_for_date(stats_dir: str, ymd: str) -> pd.DataFrame:
    """
    Read FantasyLabs NHL CSVs for a single date (YYYY-MM-DD).
    Expected files in stats_dir:
      fantasylabs_player_data_NHL_[C|W|D|G]_YYYYMMDD.csv
    """
    ymd_compact = ymd.replace("-", "")
    fps = []
    for pos in ["C","W","D","G"]:
        pat = os.path.join(stats_dir, f"fantasylabs_player_data_NHL_{pos}_{ymd_compact}.csv")
        matches = glob.glob(pat)
        if not matches:
            raise FileNotFoundError(f"Missing Labs file for position {pos}: {pat}")
        fps.extend(matches)

    parts = []
    for fp in fps:
        df = pd.read_csv(fp)
        c_player = _pick(df,"Player","Name")
        c_pos    = _pick(df,"Pos","Position")
        c_team   = _pick(df,"Team","Tm")
        c_opp    = _pick(df,"Opp","Opponent")
        c_sal    = _pick(df,"Salary","Sal")
        c_proj   = _pick(df,"Proj","Projection","My Proj","My")
        c_own    = _pick(df,"Own","Ownership","Proj Own","My Own","Ownership%","Own%")
        c_full   = _pick(df,"Full","Even Strength Line","EV Line","Line")
        c_pp     = _pick(df,"PP","Power Play","PP Unit")

        if not c_player or not c_team or not c_sal or not c_proj:
            raise ValueError(f"Missing required columns in {os.path.basename(fp)}")

        out = pd.DataFrame()
        out["Name"]   = df[c_player].astype(str)
        out["Pos"]    = df[c_pos].astype(str) if c_pos else None
        out["Team"]   = df[c_team].map(_norm_team)
        out["OppRaw"] = df[c_opp].astype(str) if c_opp else None
        out["Opp"]    = df[c_opp].map(_opp_to_team) if c_opp else None
        out["Salary"] = df[c_sal].astype(str).str.replace(r"[^0-9]", "", regex=True).map(_coerce_num)
        out["Proj"]   = pd.to_numeric(df[c_proj], errors="coerce")
        out["Own"]    = pd.to_numeric(
            df[c_own].astype(str).str.replace("%","",regex=False),
            errors="coerce"
        ) if c_own else np.nan
        out["FullRaw"]= df[c_full] if c_full else None
        out["PP"]     = df[c_pp] if c_pp else None

        # derived EV / PP features
        out["EV_Type"], out["EV_Line"] = zip(*out["FullRaw"].map(_parse_full))
        out["PP_Unit"] = out["PP"].map(_parse_pp)

        # canonical single-pos from FantasyLabs file name
        base = os.path.basename(fp).upper()
        if "_NHL_C_" in base: out["PosCanon"]="C"
        elif "_NHL_W_" in base: out["PosCanon"]="W"
        elif "_NHL_D_" in base: out["PosCanon"]="D"
        elif "_NHL_G_" in base: out["PosCanon"]="G"
        else: out["PosCanon"]=out["Pos"].str.extract(r"(C|W|D|G)")[0]

        parts.append(out)

    full = pd.concat(parts, ignore_index=True)
    # Clean: drop duplicates on (Name,Team,PosCanon) keep first
    full = full.dropna(subset=["Team","Salary","Proj","PosCanon"])
    full = full.drop_duplicates(subset=["Name","Team","PosCanon"], keep="first").reset_index(drop=True)

    # Game key for bring-backs
    def _gk(row):
        a,b = sorted([row["Team"], row["Opp"]]) if row["Opp"] else (row["Team"], None)
        return f"{a}@{b}" if b else None
    full["GameKey"] = full.apply(_gk, axis=1)

    # sanity cols
    full["IsSkater"] = full["PosCanon"].isin(["C","W","D"])
    full["IsGoalie"] = (full["PosCanon"] == "G")
    return full

DK_ROSTER = dict(C=2, W=3, D=2, G=1, UTIL=1)
DK_SALARY_CAP = 50000
