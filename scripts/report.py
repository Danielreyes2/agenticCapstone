"""Reprint the week-1 numbers from the gate logs.

The narrative is docs/REPORT.md. This command only checks that a log still
scores the way that narrative says, for whoever has the sqlite files locally.
`runs/` is gitignored, so a clone without the logs prints nothing to compare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ato.eval import evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print gate localization from a run log.")
    parser.add_argument(
        "logs",
        nargs="*",
        type=Path,
        default=[Path("runs/gate2.sqlite"), Path("runs/gate-haiku.sqlite")],
    )
    args = parser.parse_args(argv)
    missing = False
    for path in args.logs:
        if not path.exists():
            print(f"missing {path}")
            missing = True
            continue
        report = evaluate(path)
        print(path)
        for condition, rates in report.state.items():
            loc = rates.localization
            print(f"  state {condition} {loc.k}/{loc.n}")
        print(f"  schema {report.schema_validity.k}/{report.schema_validity.n}")
        print(f"  stability_mean {report.stability_mean}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
