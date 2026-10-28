"""The written report states the gate counts and the commitments it is scored against."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "REPORT.md"


def test_the_report_scores_the_commitments_and_the_gate_counts() -> None:
    text = REPORT.read_text()
    for phrase in (
        "14.2",
        "53.5",
        "temperature_sent",
        "4/5",
        "0/5",
        "5/5",
        "24/25",
        "19/25",
        "s019",
        "s033",
        "hrv_7d[2]",
        "hrv_7d[5]",
        "output",
    ):
        assert phrase in text
    assert "Output faults were not run" in text
    assert "not averaged" in text
