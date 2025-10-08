from __future__ import annotations
from typing import Dict, List, Tuple, Iterable
import pandas as pd
import numpy as np
from collections import defaultdict

def line_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach convenient line tags:
      EV_LINE_TAG: 'BOS-L1F', 'BOS-L1D', etc
      PP_TAG:      'BOS-PP1', 'BOS-PP2'
    """
    out = df.copy()
    def _lrow(r):
        if pd.notna(r.get("EV_Line")) and pd.notna(r.get("EV_Type")):
            return f"{r['Team']}-L{int(r['EV_Line'])}{str(r['EV_Type'])}"
        return None
    def _pprow(r):
        if pd.notna(r.get("PP_Unit")):
            return f"{r['Team']}-PP{int(r['PP_Unit'])}"
        return None
    out["EV_LINE_TAG"] = out.apply(_lrow, axis=1)
    out["PP_TAG"] = out.apply(_pprow, axis=1)
    return out

def group_line_members(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Return index lists for each EV line tag and PP tag.
    """
    mapping = defaultdict(list)
    for i, r in df.iterrows():
        if pd.notna(r.get("EV_LINE_TAG")):
            mapping[("EV", r["EV_LINE_TAG"])].append(i)
        if pd.notna(r.get("PP_TAG")):
            mapping[("PP", r["PP_TAG"])].append(i)
    return dict(mapping)

def game_pairs(df: pd.DataFrame) -> Dict[str, Tuple[str,str]]:
    """
    Map GameKey -> (TeamA, TeamB). Assumes df['GameKey'] is normalized 'A@B' with A<B.
    """
    out = {}
    for _,r in df.dropna(subset=["GameKey","Team","Opp"]).iterrows():
        gk = r["GameKey"]; a,b = gk.split("@")
        out[gk] = (a,b)
    return out

def count_same_team(df: pd.DataFrame, sel_mask: Iterable[bool]) -> Dict[str,int]:
    s = df.loc[sel_mask & df["IsSkater"], "Team"].value_counts()
    return s.to_dict()

# Human labels aligned to charts
STACK_LABELS = [
    "SameTeam_2+", "SameTeam_3+", "SameTeam_4+", "SameTeam_5+",
    "EVLine_2+", "EVLine_3+",
    "PP1_2+", "PP1_3+", "PP2_2+",
    "BringBack_1+"
]
