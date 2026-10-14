from __future__ import annotations

from pathlib import Path

import pytest

from ato.provider import FixtureProvider
from ato.schema import WorldState

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def clean() -> WorldState:
    """The canonical scenario. Reconciles exactly; see docs/SPEC.md."""
    return FixtureProvider(FIXTURES).load("s014")


@pytest.fixture
def all_clean() -> list[WorldState]:
    provider = FixtureProvider(FIXTURES)
    return [provider.load(sid) for sid in provider.scenario_ids()]
