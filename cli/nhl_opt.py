#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line wrapper for the NHL optimizer.

This thin CLI module:
  - Ensures the repo root is on sys.path (so imports work when running as a script)
  - Delegates straight to nhl_tools.nhl_optimizer.main (which handles argparse)
"""

from __future__ import annotations

import os
import sys
import traceback


def _ensure_repo_on_path() -> str:
    """
    Add the repository root to sys.path when running as a script.

    Returns
    -------
    str
        The absolute path to the repository root that was ensured on sys.path.
    """
    # repo_root = <repo>/  (parent of this file's directory)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def main() -> None:
    """Entry point that defers to ``nhl_tools.nhl_optimizer.main``."""
    repo_root = _ensure_repo_on_path()

    try:
        # Import here so sys.path is already fixed up
        from nhl_tools.nhl_optimizer import main as opt_main  # noqa: WPS433

    except Exception as exc:  # pragma: no cover - defensive diagnostics
        # Give a helpful diagnostic if imports blow up
        sys.stderr.write(
            "ERROR: Failed to import nhl_tools.nhl_optimizer.main\n"
            f"Repo root ensured on sys.path: {repo_root}\n"
            f"cwd: {os.getcwd()}\n"
            f"sys.executable: {sys.executable}\n"
            f"sys.path[0:3]: {sys.path[:3]}\n"
            f"Exception type: {type(exc).__name__}\n"
            f"Exception: {exc}\n"
            "Traceback:\n"
            f"{traceback.format_exc()}\n"
        )
        raise

    # Hand off to the real CLI (it will parse sys.argv)
    opt_main()


if __name__ == "__main__":
    main()
