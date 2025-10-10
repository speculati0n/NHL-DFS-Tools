import argparse
import os
import sys


def _ensure_repo_on_path() -> None:
    """Add the repository root to sys.path for direct CLI execution."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


_ensure_repo_on_path()

from nhl_tools.nhl_optimizer import main


def _parse_wrapper_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dk-upload", action="store_true", help="Also write DK-format wide CSV")
    parser.add_argument("--upload-out", help="Destination for DK upload CSV (default: <out>_DKUPLOAD.csv)")
    parser.add_argument("--player-ids", help="Path to dk_data/player_ids.csv")
    return parser.parse_known_args(argv)


def _default_player_ids_path() -> str:
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),
                        "dk_data", "player_ids.csv")


if __name__ == "__main__":
    wrapper_args, remaining = _parse_wrapper_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    out_path = main()

    if wrapper_args.dk_upload and out_path:
        ids_path = wrapper_args.player_ids or _default_player_ids_path()
        upload_out = wrapper_args.upload_out
        if not upload_out:
            base, ext = os.path.splitext(out_path)
            upload_out = base + "_DKUPLOAD.csv"
        from nhl_tools import dk_export
        dk_export.export(out_path, ids_path, upload_out, strict=False)
        print(f"[NHL][Export] wrote {upload_out}")
