import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BODY = {
    "chest": 96, "waist": 80, "hip": 102,
    "garments": [
        {"size": "S", "chest": 92, "waist": 76, "hip": 98},
        {"size": "M", "chest": 100, "waist": 84, "hip": 106},
        {"size": "L", "chest": 110, "waist": 94, "hip": 116},
    ],
    "fit_preference": "regular",
    "stretch_percent": 5,
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("GARMENT_DB_PATH", str(path))
    return path


@pytest.fixture
def client(db):
    from app.main import app
    return TestClient(app)
