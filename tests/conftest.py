import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data import load_config, load_tables  # noqa: E402
from core.planning import PlanningContext  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT)


@pytest.fixture(scope="session")
def tables(cfg):
    return load_tables(ROOT, tz=cfg["timezone"])


@pytest.fixture(scope="session")
def ctx(tables, cfg):
    return PlanningContext(tables, cfg)


@pytest.fixture
def now():
    return pd.Timestamp("2026-10-11 06:00", tz="Asia/Kolkata")
