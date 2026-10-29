import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from inventory_ai import db  # noqa: E402


@pytest.fixture()
def database(tmp_path):
    db.configure(f"sqlite:///{tmp_path / 'test.db'}")
    db.init_db()
    yield
    db.engine().dispose()


@pytest.fixture()
def session(database):
    with db.session_scope() as s:
        yield s
