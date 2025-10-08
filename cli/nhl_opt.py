import os
import sys


def _ensure_repo_on_path() -> None:
    """Add the repository root to sys.path for direct CLI execution."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_on_path()

from nhl_tools.nhl_optimizer import main
if __name__ == "__main__":
    main()
