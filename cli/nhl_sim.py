#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI wrapper for the edges-aware simulator."""
from __future__ import annotations

import os
import sys


def _ensure_repo_on_path() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> None:
    _ensure_repo_on_path()
    from nhl_tools.nhl_simulator import main as sim_main

    sim_main()


if __name__ == "__main__":
    main()
