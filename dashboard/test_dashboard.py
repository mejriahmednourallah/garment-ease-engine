import json
import urllib.error
from email.message import Message
from io import BytesIO

from dashboard import api


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self):
        return self.body


def test_build_payload_matches_contract():
    payload = api.build_payload(
        96, 80, 102,
        [{"size": " M ", "chest": 100, "waist": 84, "hip": 106}],
        "regular", 5,
    )
    assert payload == {
        "chest": 96.0, "waist": 80.0, "hip": 102.0,
        "garments": [{"size": "M", "chest": 100.0, "waist": 84.0, "hip": 106.0}],
        "fit_preference": "regular", "stretch_percent": 5.0,
    }


def test_post_json_sends_contract_json(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["content_type"] = request.get_header("Content-type")
        captured["body"] = json.loads(request.data)
        return FakeResponse(b'{"ranked": []}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert api.post_json("http://api:8000/", "/recommend", {"chest": 96}) == {"ranked": []}
    assert captured["url"] == "http://api:8000/recommend"
    assert captured["content_type"] == "application/json"
    assert captured["body"] == {"chest": 96}


def test_error_text_flattens_validation_details():
    body = json.dumps({"detail": [
        {"loc": ["body", "chest"], "msg": "Input should be less than or equal to 250"},
    ]}).encode()
    error = urllib.error.HTTPError("http://api/recommend", 422, "Unprocessable Entity", Message(), BytesIO(body))
    assert api.error_text(error) == "chest: Input should be less than or equal to 250"


def test_describe_region_uses_words_not_colour():
    tight = api.describe_region({"status": "tight", "delta_cm": -3.5})
    assert "Tight" in tight and "needs 3.5 cm more room" in tight
    assert api.describe_region({"status": "fits", "delta_cm": 0.4}) == "Fits as intended"
    assert "2.0 cm spare" in api.describe_region({"status": "loose", "delta_cm": 2.0})


def test_post_json_rejects_non_http_urls():
    import pytest
    with pytest.raises(ValueError):
        api.post_json("file:///tmp", "/secrets", {})


def test_dashboard_feedback_requires_explicit_fit_choices(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    def fake_urlopen(request, timeout):
        response = client.post("/recommend", content=request.data, headers={"Content-Type": "application/json"})
        return FakeResponse(response.content)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app_test = AppTest.from_file("streamlit_app.py").run()
    next(b for b in app_test.button if b.label == "Recommend sizes").click().run()
    assert not app_test.exception
    assert len(app_test.radio) == 3
    assert all(r.value is None for r in app_test.radio)
    next(b for b in app_test.button if b.label == "Save feedback").click().run()
    assert any("Choose the size" in error.value for error in app_test.error)


def test_research_page_shows_study_model(monkeypatch):
    from streamlit.testing.v1 import AppTest
    app_test = AppTest.from_file("pages/zhang_2026.py").run()
    assert not app_test.exception
    assert any("2026" in t.value for t in app_test.title)


def test_research_page_ranks_published_example(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    def fake_urlopen(request, timeout):
        response = client.post("/research/zhang2026/recommend", content=request.data, headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        return FakeResponse(response.content)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    at = AppTest.from_file("pages/zhang_2026.py").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.table) == 10
