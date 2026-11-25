"""Tests for the Zhang et al. (2026) jacket fit scoring framework.

Source: Zhang, X.; Yang, J.; He, H.; Qu, H.; Luo, J. An Interpretable
Multi-Dimensional Fit Evaluation Framework for Online Apparel Size
Recommendation. Textiles 2026, 6(3), 75.
https://doi.org/10.3390/textiles6030075
PDF: https://mdpi-res.com/d_attachment/textiles/textiles-06-00075/article_deploy/textiles-06-00075.pdf

Expected values below are hand calculations from Equations (2), (3), (4), (6)
and (7) and Tables 3, 4, 5, 6, 7, 8 and 10, unless marked as published Table 10
values. No participant-level records were obtained, so nothing here estimates
recommendation accuracy.
"""

import math

import pytest

from app.paper2026 import (
    SIZE_CHARTS,
    AmbiguousSegmentError,
    rank_sizes,
    score_size,
)

# Published Table 10, Participant I (Jacket A), in printed order.
# size, GEb, GEw, GEs, GEh, FEb, FEw, FEs, FEh, FSb, FSw, FSs, FSh, OFS
TABLE_10 = [
    ("M", 7.6, 12.5, 0.7, -4.0, 1.6, 6.5, 0.7, 4.0, 1.6, 3.9, 0.1, 0.8, 6.5),
    ("L", 11.6, 16.5, 1.9, -4.0, 5.6, 10.5, 1.9, 4.0, 1.1, 9.5, 0.4, 0.8, 11.8),
    ("S", 3.6, 8.5, -0.5, -9.0, 2.4, 2.5, 0.5, 9.0, 2.4, 5.3, 0.3, 9.0, 16.9),
    ("XS", -0.4, 4.5, -1.7, -9.0, 6.4, 1.5, 1.7, 9.0, 24.2, 4.5, 1.0, 9.0, 38.7),
    ("XL", 15.6, 20.5, 3.1, 1.0, 9.6, 14.5, 3.1, 1.0, 17.3, 17.4, 3.5, 0.8, 39.0),
    ("2XL", 19.6, 24.5, 4.3, 1.0, 13.6, 18.5, 4.3, 1.0, 24.5, 22.2, 6.5, 0.8, 54.0),
    ("3XL", 23.6, 28.5, 5.5, 6.0, 17.6, 22.5, 5.5, 6.0, 31.7, 33.8, 8.3, 5.4, 79.2),
    ("4XL", 27.6, 32.5, 6.7, 6.0, 21.6, 26.5, 6.7, 6.0, 38.9, 39.8, 10.1, 5.4, 94.2),
    ("5XL", 31.6, 36.5, 7.9, 11.0, 25.6, 30.5, 7.9, 11.0, 46.1, 45.8, 11.9, 12.1, 115.9),
]


def participant_i_body():
    """Recover Participant I's body from printed GE values and Table 3.

    Every row must give the same body; the fixture asserts that below.
    Height uses the size chart's height reference, not garment length.
    """
    bodies = []
    for row in TABLE_10:
        size, geb, gew, ges, geh = row[:5]
        chart = SIZE_CHARTS["A"][size]
        bodies.append(
            {
                "bust": chart["bust"] - geb,
                "waist": chart["waist"] - gew,
                "shoulder": chart["shoulder"] - ges,
                "height": chart["height_reference"] - geh,
            }
        )
    return bodies


def test_participant_i_body_is_consistent_across_all_sizes():
    bodies = participant_i_body()
    for body in bodies:
        assert body["bust"] == pytest.approx(84.4)
        assert body["waist"] == pytest.approx(73.5)
        assert body["shoulder"] == pytest.approx(39.3)
        assert body["height"] == pytest.approx(169.0)


# ---------------------------------------------------------------------------
# Hand-calculated scoring (not published values)
# ---------------------------------------------------------------------------


def test_jacket_a_size_m_hand_calculation():
    """Body from Table 10; literal Tables 4/5 and stated GIwA = 13.0.

    bust:     GE 92.0-84.4=7.6  FE |7.6-6.0|=1.6  seg3 x1.0 = 1.60
    waist:    GE 86.0-73.5=12.5 FE |12.5-13.0|=0.5 seg4 x0.9 = 0.45
    shoulder: GE 40.0-39.3=0.7  FE 0.7           seg4 x0.2 = 0.14
    height:   GE 165.0-169=-4.0 FE 4.0           seg2 x0.2 = 0.80
    OFS = 1.60 + 0.45 + 0.14 + 0.80 = 2.99
    """
    body = {"bust": 84.4, "waist": 73.5, "shoulder": 39.3, "height": 169.0}
    result = score_size("A", body, "M")

    assert result["regions"]["bust"] == pytest.approx(
        {
            "body_cm": 84.4,
            "garment_cm": 92.0,
            "ideal_ease_cm": 6.0,
            "garment_ease_cm": 7.6,
            "fit_error_cm": 1.6,
            "segment": 3,
            "segment_weight": 1.0,
            "sectional_score": 1.6,
        }
    )
    assert result["regions"]["waist"]["ideal_ease_cm"] == 13.0
    assert result["regions"]["waist"]["fit_error_cm"] == pytest.approx(0.5)
    assert result["regions"]["waist"]["segment"] == 4
    assert result["regions"]["waist"]["sectional_score"] == pytest.approx(0.45)
    assert result["regions"]["shoulder"]["segment"] == 4
    assert result["regions"]["shoulder"]["sectional_score"] == pytest.approx(0.14)
    assert result["regions"]["height"]["segment"] == 2
    assert result["regions"]["height"]["sectional_score"] == pytest.approx(0.8)
    assert result["ofs"] == pytest.approx(2.99)


def test_jacket_a_hand_calculation_second_case():
    """XS on body 80/60/36/155.

    bust:     GE 4.0, FE 2.0, seg3 x1.0 = 2.00
    waist:    GE 18.0, FE 5.0, seg5 x1.2 = 6.00
    shoulder: GE 1.6, FE 1.6, seg4 x0.2 = 0.32
    height:   GE 5.0, FE 5.0, seg4 x0.9 = 4.50
    OFS = 12.82
    """
    body = {"bust": 80.0, "waist": 60.0, "shoulder": 36.0, "height": 155.0}
    assert score_size("A", body, "XS")["ofs"] == pytest.approx(12.82)


def test_jacket_a_segment_boundaries_are_lower_inclusive_upper_exclusive():
    # GEb = 3.0 is Segment 3 (1.0), GEb = 8.0 is Segment 4 (0.2).
    at_three = score_size("A", {"bust": 89.0, "waist": 73.5, "shoulder": 39.3, "height": 169.0}, "M")
    assert at_three["regions"]["bust"]["segment"] == 3
    assert at_three["regions"]["bust"]["segment_weight"] == 1.0

    at_eight = score_size("A", {"bust": 84.0, "waist": 73.5, "shoulder": 39.3, "height": 169.0}, "M")
    assert at_eight["regions"]["bust"]["segment"] == 4
    assert at_eight["regions"]["bust"]["segment_weight"] == 0.2

    # Waist GE = -2.0 starts Segment 2 (2.1); GE = 2.0 starts Segment 3 (0.6).
    at_minus_two = score_size("A", {"bust": 84.4, "waist": 88.0, "shoulder": 39.3, "height": 169.0}, "M")
    assert at_minus_two["regions"]["waist"]["segment"] == 2
    assert at_minus_two["regions"]["waist"]["segment_weight"] == 2.1

    at_two = score_size("A", {"bust": 84.4, "waist": 84.0, "shoulder": 39.3, "height": 169.0}, "M")
    assert at_two["regions"]["waist"]["segment"] == 3
    assert at_two["regions"]["waist"]["segment_weight"] == 0.6


def test_height_uses_size_chart_height_reference():
    # M chart height 165.0; body 165.0 gives zero height ease at the GIh target.
    body = {"bust": 86.0, "waist": 73.0, "shoulder": 40.0, "height": 165.0}
    assert score_size("A", body, "M")["regions"]["height"]["garment_ease_cm"] == 0.0


# ---------------------------------------------------------------------------
# Jacket B, including the ambiguous printed shoulder segment
# ---------------------------------------------------------------------------


def test_jacket_b_size_m_hand_calculation_with_unambiguous_shoulder():
    """B/M on bust 85.5, waist 72.0, shoulder 34.5, height 160.0.

    bust:     GE 4.5, GI 4.5, FE 0.0, seg3 x0.2 = 0.0
    waist:    GE 5.0, GI 5.0, FE 0.0, seg3 x0.8 = 0.0
    shoulder: GE 4.0, GI 0.0, FE 4.0, seg6 x1.5 = 6.0
    height:   GE 0.0, GI 0.0, FE 0.0, seg3 x0.8 = 0.0
    OFS = 6.0
    """
    body = {"bust": 85.5, "waist": 72.0, "shoulder": 34.5, "height": 160.0}
    result = score_size("B", body, "M")
    assert result["regions"]["bust"]["segment"] == 3
    assert result["regions"]["bust"]["sectional_score"] == 0.0
    assert result["regions"]["waist"]["segment"] == 3
    assert result["regions"]["shoulder"]["segment"] == 6
    assert result["regions"]["shoulder"]["sectional_score"] == pytest.approx(6.0)
    assert result["ofs"] == pytest.approx(6.0)


def test_jacket_b_shoulder_overlap_fails_closed_by_default():
    # B/M shoulder 38.5 - 39.5 = -1.0, inside the printed overlap [-2.0, 2.0).
    body = {"bust": 85.5, "waist": 72.0, "shoulder": 39.5, "height": 160.0}
    with pytest.raises(AmbiguousSegmentError) as excinfo:
        score_size("B", body, "M")
    assert "shoulder" in str(excinfo.value)
    assert isinstance(excinfo.value, ValueError)


def test_jacket_b_shoulder_overlap_can_be_explicitly_resolved():
    body = {"bust": 85.5, "waist": 72.0, "shoulder": 39.5, "height": 160.0}

    as_segment_two = score_size("B", body, "M", b_shoulder_rule="printed_segment2")
    assert as_segment_two["regions"]["shoulder"]["segment"] == 2
    assert as_segment_two["regions"]["shoulder"]["segment_weight"] == 0.5
    assert as_segment_two["regions"]["shoulder"]["sectional_score"] == pytest.approx(0.5)
    assert as_segment_two["ofs"] == pytest.approx(0.5)

    as_specific = score_size("B", body, "M", b_shoulder_rule="printed_segments3_4")
    assert as_specific["regions"]["shoulder"]["segment"] == 3
    assert as_specific["regions"]["shoulder"]["segment_weight"] == 0.2
    assert as_specific["regions"]["shoulder"]["sectional_score"] == pytest.approx(0.2)
    assert as_specific["ofs"] == pytest.approx(0.2)


def test_jacket_b_shoulder_overlap_boundaries():
    # -2.0 is inside the overlap; 2.0 is not (Segment 5 starts at 2.0).
    inside = {"bust": 85.5, "waist": 72.0, "shoulder": 40.5, "height": 160.0}
    with pytest.raises(AmbiguousSegmentError):
        score_size("B", inside, "M")

    outside = {"bust": 85.5, "waist": 72.0, "shoulder": 36.5, "height": 160.0}
    result = score_size("B", outside, "M")
    assert result["regions"]["shoulder"]["garment_ease_cm"] == 2.0
    assert result["regions"]["shoulder"]["segment"] == 5
    assert result["regions"]["shoulder"]["segment_weight"] == 0.7


# ---------------------------------------------------------------------------
# Ranking and validation
# ---------------------------------------------------------------------------


def test_participant_i_ranking_top_three_matches_published_order():
    body = participant_i_body()[0]
    ranked = rank_sizes("A", body)
    assert [item["size"] for item in ranked[:3]] == ["M", "L", "S"]
    assert [round(item["ofs"], 2) for item in ranked] == [
        2.99,
        12.10,
        14.40,
        32.74,
        39.44,
        48.98,
        68.58,
        83.58,
        105.28,
    ]


def test_ranking_is_ascending_and_chart_size_set_is_complete():
    # Body sits exactly on every Jacket A ideal for size M (OFS 0).
    body = {"bust": 86.0, "waist": 73.0, "shoulder": 40.0, "height": 165.0}
    ranked = rank_sizes("A", body)
    assert [item["ofs"] for item in ranked] == sorted(item["ofs"] for item in ranked)
    assert [item["size"] for item in ranked] == ["M", "L", "S", "XL", "XS", "2XL", "3XL", "4XL", "5XL"]


@pytest.mark.parametrize(
    "jacket, body, size, rule",
    [
        ("C", {"bust": 86, "waist": 73, "shoulder": 40, "height": 165}, "M", "error"),
        ("A", {"bust": 86, "waist": 73, "shoulder": 40, "height": 165}, "XXS", "error"),
        ("A", {"bust": 86, "waist": 73, "shoulder": 40}, "M", "error"),
        ("A", {"bust": math.nan, "waist": 73, "shoulder": 40, "height": 165}, "M", "error"),
        ("A", {"bust": 86, "waist": 73, "shoulder": 40, "height": 165}, "M", "bogus"),
    ],
)
def test_invalid_inputs_raise_value_error(jacket, body, size, rule):
    with pytest.raises(ValueError):
        score_size(jacket, body, size, b_shoulder_rule=rule)


# ---------------------------------------------------------------------------
# Comparison with the only published participant example (Table 10)
#
# These tests quantify the print inconsistencies; they do not repair them.
# ---------------------------------------------------------------------------


def test_table10_garment_ease_is_reproduced_for_all_nine_sizes():
    body = participant_i_body()[0]
    for row in TABLE_10:
        size = row[0]
        regions = score_size("A", body, size)["regions"]
        assert regions["bust"]["garment_ease_cm"] == pytest.approx(row[1])
        assert regions["waist"]["garment_ease_cm"] == pytest.approx(row[2])
        assert regions["shoulder"]["garment_ease_cm"] == pytest.approx(row[3])
        assert regions["height"]["garment_ease_cm"] == pytest.approx(row[4])


def test_table10_bust_and_shoulder_and_height_errors_match():
    body = participant_i_body()[0]
    for row in TABLE_10:
        size = row[0]
        regions = score_size("A", body, size)["regions"]
        assert regions["bust"]["fit_error_cm"] == pytest.approx(row[5])
        assert regions["shoulder"]["fit_error_cm"] == pytest.approx(row[7])
        assert regions["height"]["fit_error_cm"] == pytest.approx(row[8])


def test_table10_waist_error_column_uses_gi_6_not_stated_gi_13():
    """Published FEw equals |GEw - 6.0| for every row, not |GEw - 13.0|.

    The paper states GIwA = 13.0 cm, which is what this module implements.
    """
    body = participant_i_body()[0]
    for row in TABLE_10:
        size, _, gew, _, _, _, published_few, _, _, _, _, _, _, _ = row
        assert published_few == pytest.approx(abs(gew - 6.0))
        computed = score_size("A", body, size)["regions"]["waist"]["fit_error_cm"]
        assert computed == pytest.approx(abs(gew - 13.0))
        assert computed != pytest.approx(published_few)


def test_table10_section_scores_that_match_and_do_not_match():
    body = participant_i_body()[0]
    matching_shoulder_height = 0
    matching_bust = 0
    matching_waist = 0
    for row in TABLE_10:
        size = row[0]
        regions = score_size("A", body, size)["regions"]
        if regions["shoulder"]["sectional_score"] == pytest.approx(row[11], abs=0.1):
            matching_shoulder_height += 1
        if regions["height"]["sectional_score"] == pytest.approx(row[12], abs=0.1):
            matching_shoulder_height += 1
        if regions["bust"]["sectional_score"] == pytest.approx(row[9], abs=0.15):
            matching_bust += 1
        if regions["waist"]["sectional_score"] == pytest.approx(row[10], abs=0.15):
            matching_waist += 1

    # Shoulder (9 rows) + height (9 rows) match at 1-dp precision.
    assert matching_shoulder_height == 18
    # Bust matches except L, where GE 11.6 is Segment 5 (x1.2) but the printed
    # FSb 1.1 = 5.6 x 0.2 is the Segment 4 coefficient.
    assert matching_bust == 8
    assert matching_waist == 0


def test_table10_ofs_differences_are_quantified_without_claiming_reproduction():
    body = participant_i_body()[0]
    diffs = []
    for row in TABLE_10:
        size = row[0]
        computed = score_size("A", body, size)["ofs"]
        diffs.append(computed - row[13])

    # Differences are driven by the waist FE column using GI 6.0 instead of 13.0.
    assert diffs == pytest.approx(
        [2.99 - 6.5, 12.10 - 11.8, 14.40 - 16.9, 39.44 - 38.7, 32.74 - 39.0,
         48.98 - 54.0, 68.58 - 79.2, 83.58 - 94.2, 105.28 - 115.9]
    )
    assert max(abs(diff) for diff in diffs) == pytest.approx(10.62)


def test_table10_top_three_order_matches_despite_ofs_differences():
    body = participant_i_body()[0]
    ranked = rank_sizes("A", body)
    assert [item["size"] for item in ranked[:3]] == ["M", "L", "S"]
