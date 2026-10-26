"""Run one scenario and append one row to the log."""

from __future__ import annotations

import argparse
from pathlib import Path

from ato.runner import execute


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one ATO scenario and log the row.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--fault", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--db", type=Path, default=Path("runs.sqlite"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/llm"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    model = args.model
    if args.dry_run and model is None:
        model = "fake"
    if model is None:
        parser.error("--model is required unless --dry-run")
    execute(
        scenario_id=args.scenario,
        condition=args.condition,
        fault_id=args.fault,
        seed=args.seed,
        temperature=args.temperature,
        model=model,
        db_path=args.db,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
