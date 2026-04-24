"""Main entrypoint for the modular project CLI."""

from __future__ import annotations

import sys

from ainee_politics.presentation.cli import run_cli


if __name__ == "__main__":
    raise SystemExit(run_cli(sys.argv[1:]))