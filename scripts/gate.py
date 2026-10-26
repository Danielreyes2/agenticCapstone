"""Run the week-1 gate and print the four numbers from the log."""

from __future__ import annotations

import argparse
from pathlib import Path

from ato.gate import MODEL, format_report, run_gate, watch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Week-1 blame gate. Pilot first.")
    parser.add_argument("--pilot", action="store_true", help="one clean s014, seed 1")
    parser.add_argument("--confirm", action="store_true", help="allow the other live runs")
    parser.add_argument(
        "--d",
        action="store_true",
        help="condition D on the five faulted scenarios",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="follow the log of a run already going",
    )
    # runs/gate.sqlite and .cache/gate hold the first gate and are kept as evidence.
    parser.add_argument("--db", type=Path, default=Path("runs/gate2.sqlite"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/gate2"))
    args = parser.parse_args(argv)
    if args.watch:
        watch(args.db)
        return 0
    report = run_gate(
        db_path=args.db,
        cache_dir=args.cache_dir,
        dry_run=args.dry_run,
        pilot=args.pilot,
        confirm=args.confirm,
        rescue=args.d,
        show_progress=True,
    )
    print(f"model {MODEL if not args.dry_run else 'fake'}")
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
