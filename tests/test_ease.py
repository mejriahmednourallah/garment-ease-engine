import pytest

from app.main import rank_sizes


def test_closest_size_wins():
    body = {"chest": 100, "waist": 84, "hip": 106}
    garments = [
        {"size": "S", "chest": 90, "waist": 74, "hip": 96},
        {"size": "M", "chest": 108, "waist": 89, "hip": 114},
    ]
    ranked = rank_sizes(body, garments, "regular", 0)
    assert [r["size"] for r in ranked] == ["M", "S"]
    # regular targets 8/5/8 cm: M sits exactly on them, S is far below
    assert ranked[0]["score"] == pytest.approx(0.0)


def test_stretch_reduces_required_size():
    body = {"chest": 100, "waist": 84, "hip": 106}
    garment = [{"size": "S", "chest": 92, "waist": 76, "hip": 98}]
    stiff = rank_sizes(body, garment, "regular", 0)[0]
    stretchy = rank_sizes(body, garment, "regular", 10)[0]
    assert stretchy["score"] < stiff["score"]
    # 10% of a 100 cm chest is 10 cm of give: required 100 + 8 - 10 = 98, garment 92
    assert stretchy["regions"]["chest"]["delta_cm"] == pytest.approx(-6)


def test_region_explanation_status():
    body = {"chest": 100, "waist": 84, "hip": 106}
    garment = [{"size": "X", "chest": 90, "waist": 89, "hip": 120}]
    regions = rank_sizes(body, garment, "regular", 0)[0]["regions"]
    assert regions["chest"]["status"] == "tight"
    assert regions["waist"]["status"] == "fits"
    assert regions["hip"]["status"] == "loose"


def test_weights_sum_to_one():
    body = {"chest": 100, "waist": 84, "hip": 106}
    ranked = rank_sizes(body, [{"size": "M", "chest": 100, "waist": 84, "hip": 106}], "slim", 0)
    regions = ranked[0]["regions"]
    total = sum(r["delta_cm"] ** 2 * w for r, w in zip(regions.values(), (0.4, 0.3, 0.3)))
    assert ranked[0]["score"] == pytest.approx(round(total, 2), abs=0.01)


def test_ranking_uses_unrounded_scores_for_near_ties():
    body = {"chest": 100, "waist": 84, "hip": 106}
    garments = [
        {"size": "A", "chest": 108.10, "waist": 89, "hip": 114},
        {"size": "Z", "chest": 108.01, "waist": 89, "hip": 114},
    ]
    assert [item["size"] for item in rank_sizes(body, garments, "regular", 0)] == ["Z", "A"]
