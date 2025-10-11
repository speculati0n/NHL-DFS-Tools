#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


def _ensure_repo_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_on_path()


def main():
    from nhl_tools.nhl_optimizer import main as opt_main

    opt_main()


if __name__ == "__main__":
    main()
