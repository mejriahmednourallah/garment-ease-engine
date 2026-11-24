"""HTTP client and payload mapping for the garment dashboard. Standard library only."""

import json
import urllib.request

REGIONS = ("chest", "waist", "hip")
FIT_OPTIONS = ("slim", "regular", "loose")


def build_payload(chest, waist, hip, garments, fit_preference, stretch_percent):
    """Map the form state onto the POST /recommend body from CONTRACT.md."""
    return {
        "chest": float(chest),
        "waist": float(waist),
        "hip": float(hip),
        "garments": [
            {
                "size": garment["size"].strip(),
                "chest": float(garment["chest"]),
                "waist": float(garment["waist"]),
                "hip": float(garment["hip"]),
            }
            for garment in garments
        ],
        "fit_preference": fit_preference,
        "stretch_percent": float(stretch_percent),
    }


def post_json(base_url, path, payload, timeout=15):
    """POST payload as JSON and return the decoded response body."""
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def error_text(error):
    """Readable message for an HTTPError, flattening FastAPI 422 details."""
    try:
        detail = json.loads(error.read().decode("utf-8")).get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, list):
        message = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', ()) if part != 'body')}: "
            f"{item.get('msg', 'invalid')}"
            for item in detail
        )
        if message:
            return message
    if detail:
        return str(detail)
    return f"The API returned HTTP {error.code}."


def describe_region(region):
    """Plain-language result for one region; status is never colour-only."""
    delta = region["delta_cm"]
    if region["status"] == "tight":
        return f"Tight — needs {abs(delta):.1f} cm more room"
    if region["status"] == "loose":
        return f"Loose — {delta:.1f} cm spare"
    return "Fits as intended"
