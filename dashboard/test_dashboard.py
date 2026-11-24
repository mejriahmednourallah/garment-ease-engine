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
