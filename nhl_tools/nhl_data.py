# nhl_tools/nhl_stacks.py
import re
from typing import Optional
import pandas as pd

def _to_int_or_none(x) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().upper()
    m = re.search(r"(\d+)", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    try:
        return int(float(s))
    except Exception:
        return None

def line_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize EV/PP fields and build stack tags for skaters:
      EV_TAG = TEAM-L{1..4}, PP_TAG = TEAM-PP{1..2}
    """
    out = df.copy()

    if "EV_Line" in out.columns:
        out["EV_Line"] = out["EV_Line"].map(_to_int_or_none)
    else:
        out["EV_Line"] = None

    if "PP_Unit" in out.columns:
        out["PP_Unit"] = out["PP_Unit"].map(_to_int_or_none)
    else:
        out["PP_Unit"] = None

    if "PosCanon" not in out.columns:
        out["PosCanon"] = None
    is_skater = out["PosCanon"].isin(["C", "W", "D"])

    def _evrow(r):
        if not is_skater.iloc[r.name]:
            return None
        if pd.isna(r.get("Team")) or pd.isna(r.get("EV_Line")):
            return None
        return f"{r['Team']}-L{int(r['EV_Line'])}"

    def _pprow(r):
        if not is_skater.iloc[r.name]:
            return None
        if pd.isna(r.get("Team")) or pd.isna(r.get("PP_Unit")):
            return None
        return f"{r['Team']}-PP{int(r['PP_Unit'])}"

    out["EV_TAG"] = out.apply(_evrow, axis=1)
    out["PP_TAG"] = out.apply(_pprow, axis=1)
    return out
