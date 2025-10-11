#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command line wrapper for the NHL optimizer."""
from __future__ import annotations

import os
import sys


def _ensure_repo_on_path() -> None:
    """Add the repository root to ``sys.path`` when running as a script."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> None:
    """Entry-point that defers to ``nhl_tools.nhl_optimizer``."""
    _ensure_repo_on_path()

    from nhl_tools.nhl_optimizer import main as opt_main

    opt_main()


if __name__ == "__main__":
    main()
