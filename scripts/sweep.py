"""Run the condition sweep and print the harness report.

Condition A, the open-weight model, and the temperature sweep are not in this
command. See `ato/sweep.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ato.generate import SWEEP_COUNT
from ato.sweep import MODEL, format_sweep, run_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Condition sweep. Dry-run first.")
    parser.add_argument("--confirm", action="store_true", help="allow the live sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--count", type=int, default=SWEEP_COUNT)
    parser.add_argument("--seed", type=int, default=0, help="seed for the scenario generator")
    parser.add_argument("--db", type=Path, default=Path("runs/sweep.sqlite"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/sweep"))
    parser.add_argument("--scenarios-dir", type=Path, default=Path("runs/sweep-scenarios"))
    parser.add_argument("--model", default=MODEL, help="live model id; default stays Opus")
    args = parser.parse_args(argv)
    result = run_sweep(
        count=args.count,
        scenario_seed=args.seed,
        db_path=args.db,
        cache_dir=args.cache_dir,
        scenarios_dir=args.scenarios_dir,
        dry_run=args.dry_run,
        confirm=args.confirm,
        show_progress=True,
        model=args.model,
    )
    print(f"model {args.model if not args.dry_run else 'fake'}")
    print(format_sweep(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
