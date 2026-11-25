"""Zhang et al. (2026) jacket fit scoring: ideal ease, fit error, piecewise weights, OFS.

Source: Zhang, X.; Yang, J.; He, H.; Qu, H.; Luo, J. An Interpretable
Multi-Dimensional Fit Evaluation Framework for Online Apparel Size
Recommendation. Textiles 2026, 6(3), 75.
DOI: https://doi.org/10.3390/textiles6030075
PDF: https://mdpi-res.com/d_attachment/textiles/textiles-06-00075/article_deploy/textiles-06-00075.pdf
License: CC BY 4.0 (article). Tables transcribed, not redistributed.

Implementation contract
-----------------------
Inputs: jacket ``"A"`` (regular fit, nine sizes) or ``"B"`` (tight fit, seven
sizes); body measurements in cm with keys ``bust``, ``waist``, ``shoulder``,
``height``; a size label from that jacket's published chart.

For each region i in bust, waist, shoulder, height:

    GE_i = garment_cm_i - body_cm_i                              (Eq. 2)
    FE_i = |GE_i - GI_i|                                         (Eq. 3/4)
    FS_i = W_ij * FE_i, where GE_i lies in segment P_ij          (Eq. 6)
    OFS  = sum_i FS_i                                           (Eq. 7)

``score_size`` returns ``{"jacket", "size", "regions", "ofs"}``. Each region
holds ``body_cm``, ``garment_cm``, ``ideal_ease_cm``, ``garment_ease_cm``,
``fit_error_cm``, 1-based ``segment``, ``segment_weight`` and
``sectional_score``. Values are unrounded. ``rank_sizes`` returns all chart
sizes sorted by ascending unrounded OFS, ties by published chart order.

Segment intervals are lower-inclusive, upper-exclusive, as printed. The last
segment is open-ended. There are no region-level weights: OFS is the plain
sum of the four sectional scores.

Documented ambiguities and print inconsistencies (not repaired)
---------------------------------------------------------------
1. Equation (4) uses the absolute difference between actual and ideal ease.
   Some plain-text PDF extractors omit the absolute-value bars.
2. Height: Table 3/6 label the height column "Human Fit Model ... Height".
   Table 10's height ease is exactly that chart height minus body height
   (M: 165 - 169 = -4.0), whereas garment length would give 63 - 169 = -106.
   Section 3.3 calls height a size-chart reference index, so height ease here
   is ``chart["height_reference"] - body["height"]``. Garment length is not
   used for scoring.
3. Table 10's waist FE column equals |GEw - 6.0| on all nine rows, not
   |GEw - 13.0| as the stated GIwA = 13.0 requires. This module uses 13.0;
   tests quantify the resulting mismatches. Table 10 publishes no body
   measurements; the tests recover Participant I as 84.4/73.5/39.3/169.0 cm
   from the printed GE values and Table 3, and assert all nine rows agree.
   The computed OFS differs from Table 10 on all nine sizes (largest
   difference 10.62); the Top-3 order still comes out M, L, S.
4. Table 10's L row prints FSb = 1.1, which is 5.6 x 0.2 (Segment 4 weight),
   although GEb = 11.6 lies in Segment 5 (weight 1.2). Tables 4/5 are used
   literally.
5. Table 4's waist intervals put the stated ideal GIwA = 13.0 in Segment 4
   (weight 0.9), not the smallest-weight segment, and its FSw column cannot
   be reproduced under any single consistent reading of Tables 4/5. Tables
   are used literally; Table 10's FSw values do not match this module. The
   stated "smallest weight contains GI" principle also fails for A bust
   (GI 6.0 in Segment 3, weight 1.0, while Segment 4 weighs 0.2) and A
   height (GI 0.0 in Segment 3, weight 0.8, while Segment 2 weighs 0.2).
   Calibrated weights are kept as printed.
6. Table 7 prints Jacket B's shoulder Segment 2 as -4.0 <= s < 2.0, which
   overlaps Segment 3 (-2.0 <= s < 0.0) and Segment 4 (0.0 <= s < 2.0).
   For GE in [-2.0, 2.0) scoring fails closed with AmbiguousSegmentError
   unless the caller explicitly chooses ``b_shoulder_rule``:
   ``"printed_segment2"`` (use its 0.5 weight) or ``"printed_segments3_4"``
   (use the more specific 0.2 / 0.4 weights). Neither choice is validated by
   the paper.
7. Table 6's L fit-model bust 85.5 cm conflicts with the stated GIbB = 4.5 cm
   and the other six rows. Stated ideal-ease constants are used; the chart's
   fit-model columns are not used for scoring.

No participant-level records were obtained. Only one worked example (Table 10,
Participant I) is published. Nothing here validates the paper's reported
Top-1/Top-2/Top-3 accuracy, and no accuracy is claimed.
"""

import math

REGIONS = ("bust", "waist", "shoulder", "height")

SOURCE_DOI = "https://doi.org/10.3390/textiles6030075"
SOURCE_PDF = (
    "https://mdpi-res.com/d_attachment/textiles/textiles-06-00075/"
    "article_deploy/textiles-06-00075.pdf"
)

# Garment specifications from Tables 3 and 6. "height_reference" is the
# published "Human Fit Model" height for the size (see ambiguity note 2).
SIZE_CHARTS = {
    "A": {
        "XS": {"bust": 84.0, "waist": 78.0, "shoulder": 37.6, "height_reference": 160.0},
        "S": {"bust": 88.0, "waist": 82.0, "shoulder": 38.8, "height_reference": 160.0},
        "M": {"bust": 92.0, "waist": 86.0, "shoulder": 40.0, "height_reference": 165.0},
        "L": {"bust": 96.0, "waist": 90.0, "shoulder": 41.2, "height_reference": 165.0},
        "XL": {"bust": 100.0, "waist": 94.0, "shoulder": 42.4, "height_reference": 170.0},
        "2XL": {"bust": 104.0, "waist": 98.0, "shoulder": 43.6, "height_reference": 170.0},
        "3XL": {"bust": 108.0, "waist": 102.0, "shoulder": 44.8, "height_reference": 175.0},
        "4XL": {"bust": 112.0, "waist": 106.0, "shoulder": 46.0, "height_reference": 175.0},
        "5XL": {"bust": 116.0, "waist": 110.0, "shoulder": 47.2, "height_reference": 180.0},
    },
    "B": {
        "S": {"bust": 86.0, "waist": 73.0, "shoulder": 37.5, "height_reference": 155.0},
        "M": {"bust": 90.0, "waist": 77.0, "shoulder": 38.5, "height_reference": 160.0},
        "L": {"bust": 94.0, "waist": 81.0, "shoulder": 39.5, "height_reference": 160.0},
        "XL": {"bust": 98.0, "waist": 85.0, "shoulder": 40.5, "height_reference": 165.0},
        "2XL": {"bust": 102.0, "waist": 89.0, "shoulder": 41.5, "height_reference": 170.0},
        "3XL": {"bust": 106.0, "waist": 93.0, "shoulder": 42.5, "height_reference": 170.0},
        "4XL": {"bust": 110.0, "waist": 97.0, "shoulder": 43.5, "height_reference": 175.0},
    },
}

# Stated ideal ease, Section 4.2.2 (Jacket A) and 4.2.3 (Jacket B), in cm.
IDEAL_EASE = {
    "A": {"bust": 6.0, "waist": 13.0, "shoulder": 0.0, "height": 0.0},
    "B": {"bust": 4.5, "waist": 5.0, "shoulder": 0.0, "height": 0.0},
}

# Tables 4 and 7: per region, (exclusive upper bound, weight), in order.
# Segment number is the 1-based position; None means open-ended.
SEGMENTS = {
    "A": {
        "bust": ((0.0, 3.8), (3.0, 3.5), (8.0, 1.0), (11.0, 0.2), (14.0, 1.2), (None, 1.8)),
        "waist": ((-2.0, 3.0), (2.0, 2.1), (10.0, 0.6), (15.0, 0.9), (20.0, 1.2), (None, 1.5)),
        "shoulder": ((-4.0, 1.0), (-2.0, 0.8), (0.0, 0.6), (2.0, 0.2), (4.0, 1.1), (None, 1.5)),
        "height": ((-5.0, 1.0), (0.0, 0.2), (5.0, 0.8), (10.0, 0.9), (15.0, 1.1), (None, 1.2)),
    },
    "B": {
        "bust": ((0.0, 2.5), (2.0, 1.8), (6.0, 0.2), (10.0, 0.9), (14.0, 1.2), (None, 1.4)),
        "waist": ((-5.0, 3.0), (5.0, 0.5), (10.0, 0.8), (15.0, 0.9), (20.0, 1.2), (None, 1.5)),
        # Printed as -4.0 <= s < 2.0; overlaps Segments 3 and 4 (note 6).
        "shoulder": ((-4.0, 1.0), (2.0, 0.5), (0.0, 0.2), (2.0, 0.4), (4.0, 0.7), (None, 1.5)),
        "height": ((-10.0, 1.0), (0.0, 0.2), (5.0, 0.8), (10.0, 0.9), (15.0, 1.1), (None, 1.2)),
    },
}

B_SHOULDER_RULES = ("error", "printed_segment2", "printed_segments3_4")


class AmbiguousSegmentError(ValueError):
    """Table 7 assigns two different weights to Jacket B's shoulder ease."""


def _check_jacket(jacket):
    if jacket not in SIZE_CHARTS:
        raise ValueError(f"jacket must be 'A' or 'B', got {jacket!r}")


def _check_rule(rule):
    if rule not in B_SHOULDER_RULES:
        raise ValueError(f"b_shoulder_rule must be one of {B_SHOULDER_RULES}, got {rule!r}")


def _check_body(body):
    for region in REGIONS:
        value = body.get(region) if hasattr(body, "get") else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"body.{region} must be a finite positive number in cm")


def _segment(jacket, region, ease, b_shoulder_rule):
    if jacket == "B" and region == "shoulder" and -2.0 <= ease < 2.0:
        if b_shoulder_rule == "error":
            raise AmbiguousSegmentError(
                "Jacket B shoulder ease "
                f"{ease:g} cm lies in the printed overlap [-2.0, 2.0): Table 7 "
                "Segment 2 (-4.0 <= s < 2.0) overlaps Segments 3 and 4. Pass "
                "b_shoulder_rule='printed_segment2' or 'printed_segments3_4' "
                "to choose a reading explicitly."
            )
        if b_shoulder_rule == "printed_segment2":
            return 2, 0.5
        # Segment 3 is [-2.0, 0.0), Segment 4 is [0.0, 2.0).
        return (3, 0.2) if ease < 0 else (4, 0.4)

    for index, (upper, weight) in enumerate(SEGMENTS[jacket][region]):
        if upper is None or ease < upper:
            return index + 1, weight
    raise AssertionError("open-ended segment missing")


def score_size(jacket, body, size, *, b_shoulder_rule="error"):
    """Score one candidate size with the published Zhang et al. (2026) tables."""
    _check_jacket(jacket)
    _check_rule(b_shoulder_rule)
    chart = SIZE_CHARTS[jacket]
    if size not in chart:
        raise ValueError(f"size must be one of {tuple(chart)}, got {size!r}")
    _check_body(body)

    regions = {}
    ofs = 0.0
    for region in REGIONS:
        garment_cm = chart[size]["height_reference"] if region == "height" else chart[size][region]
        ideal = IDEAL_EASE[jacket][region]
        garment_ease = garment_cm - body[region]
        fit_error = abs(garment_ease - ideal)
        segment, weight = _segment(jacket, region, garment_ease, b_shoulder_rule)
        sectional = weight * fit_error
        ofs += sectional
        regions[region] = {
            "body_cm": float(body[region]),
            "garment_cm": garment_cm,
            "ideal_ease_cm": ideal,
            "garment_ease_cm": garment_ease,
            "fit_error_cm": fit_error,
            "segment": segment,
            "segment_weight": weight,
            "sectional_score": sectional,
        }

    return {"jacket": jacket, "size": size, "regions": regions, "ofs": ofs}


def rank_sizes(jacket, body, *, b_shoulder_rule="error"):
    """Score every published size, ascending OFS, ties in chart order."""
    _check_jacket(jacket)
    order = {size: index for index, size in enumerate(SIZE_CHARTS[jacket])}
    ranked = [
        score_size(jacket, body, size, b_shoulder_rule=b_shoulder_rule)
        for size in order
    ]
    ranked.sort(key=lambda item: (item["ofs"], order[item["size"]]))
    return ranked
