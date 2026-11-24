import json
import sqlite3

import pytest

from tests.conftest import BODY


def test_recommend_ranks_and_explains(client):
    r = client.post("/recommend", json=BODY)
    assert r.status_code == 200
    data = r.json()
    assert data["fit_preference"] == "regular"
    assert data["weights"] == {"chest": 0.4, "waist": 0.3, "hip": 0.3}
    sizes = [x["size"] for x in data["ranked"]]
    assert len(sizes) == len(BODY["garments"])
    assert sizes[0] == "M"
    scores = [x["score"] for x in data["ranked"]]
    assert scores == sorted(scores)
    chest = data["ranked"][0]["regions"]["chest"]
    assert set(chest) == {
        "body_cm", "garment_cm", "target_ease_cm",
        "actual_ease_cm", "delta_cm", "status",
    }
    assert chest["actual_ease_cm"] == pytest.approx(4)
    # 5% stretch on 96 cm gives 4.8 cm, so M is within the 2 cm band of the 8 cm target
    assert chest["status"] == "fits"


def test_recommend_validation(client):
    bad = [
        {**BODY, "chest": 10},                       # out of range
        {**BODY, "chest": "abc"},                    # wrong type
        {**BODY, "fit_preference": "oversized"},     # unknown preference
        {**BODY, "stretch_percent": 80},             # out of range
        {**BODY, "garments": []},                    # nothing to rank
        {**BODY, "garments": [{"size": "", "chest": 100, "waist": 84, "hip": 106}]},
        {k: v for k, v in BODY.items() if k != "waist"},  # missing body measure
    ]
    for payload in bad:
        assert client.post("/recommend", json=payload).status_code == 422, payload


def test_feedback_persists_to_configured_db(client, db):
    payload = {"request": BODY, "size": "M",
               "actual_fit": {"chest": "tight", "waist": "fits", "hip": "fits"}}
    r = client.post("/feedback", json=payload)
    assert r.status_code == 201
    assert r.json()["id"] == 1
    assert r.json()["created_at"]

    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT size, fit_preference, stretch_percent, request_json, actual_fit_json FROM feedback"
        ).fetchone()
    assert row[0] == "M"
    assert row[1] == "regular"
    assert row[2] == 5
    assert json.loads(row[3])["garments"][1]["size"] == "M"
    assert json.loads(row[4]) == payload["actual_fit"]


def test_feedback_rejects_unknown_size_and_bad_fit(client, db):
    good = {"request": BODY, "size": "M",
            "actual_fit": {"chest": "fits", "waist": "fits", "hip": "fits"}}
    assert client.post("/feedback", json=good).status_code == 201
    bad = [
        {"request": BODY, "size": "XL", "actual_fit": {"chest": "fits", "waist": "fits", "hip": "fits"}},
        {"request": BODY, "size": "M", "actual_fit": {"chest": "snug"}},
    ]
    for payload in bad:
        assert client.post("/feedback", json=payload).status_code == 422, payload
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1


def test_db_path_env_is_honoured(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app

    c = TestClient(app)
    payload = {"request": BODY, "size": "S",
               "actual_fit": {"chest": "fits", "waist": "tight", "hip": "fits"}}
    for name in ("a.db", "b.db"):
        monkeypatch.setenv("GARMENT_DB_PATH", str(tmp_path / name))
        assert c.post("/feedback", json=payload).status_code == 201
        with sqlite3.connect(tmp_path / name) as conn:
            assert conn.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1


def test_rejects_blank_or_duplicate_size_labels(client):
    from conftest import BODY
    import copy
    payload = copy.deepcopy(BODY)
    payload["garments"][0]["size"] = "  "
    assert client.post("/recommend", json=payload).status_code == 422
    payload = copy.deepcopy(BODY)
    payload["garments"][1]["size"] = payload["garments"][0]["size"]
    assert client.post("/recommend", json=payload).status_code == 422


def test_nonfinite_json_returns_validation_error(client):
    from conftest import BODY
    import json
    payload = dict(BODY, chest=float("nan"))
    response = client.post("/recommend", content=json.dumps(payload), headers={"Content-Type": "application/json"})
    assert response.status_code == 422
