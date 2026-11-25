#!/usr/bin/env python3
"""Extract Dik et al. (2023) Appendix 2 pattern-parameter ease table into CSV.

Source: Heliyon 9(7):e17916, Appendix 2 "Ease Allowance Preference Charts for Each
Garment Type". The article publishes this one numeric table; the per-size ease
"charts" themselves are inline images and are not machine-readable.

These are *pattern-parameter adjustments in cm* preferred by each of five
psychographic clusters (P1 lowest AIO involvement ... P5 highest), per garment
type. They are NOT full-circumference garment ease and must not be substituted
directly for a regional ease target. Aggregate values published by the authors;
not participant-level data.

Run:  python extract_dik2023_pattern_table.py
Writes: dik_2023_appendix2_pattern_ease_by_group.csv
"""
from __future__ import annotations

import csv
import hashlib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

PMC_XML = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10362334/fullTextXML"
OUT = Path(__file__).resolve().parent / "dik_2023_appendix2_pattern_ease_by_group.csv"


def main() -> int:
    raw = urllib.request.urlopen(PMC_XML, timeout=60).read()
    print(f"fetched {len(raw):,} bytes, sha256 {hashlib.sha256(raw).hexdigest()[:16]}...")
    root = ET.fromstring(raw)

    appendix = next(s for s in root.iter("sec")
                    if (s.findtext("title") or "").strip().lower() == "appendix")
    table = next(iter(appendix.iter("table-wrap")))

    rows, garment = [], None
    for tr in table.iter("tr"):
        cells = [" ".join(" ".join(td.itertext()).split()) for td in tr.iter("td")]
        if not cells:
            continue
        if len(cells) == 1:                       # garment-type header row
            garment = cells[0]
            continue
        if len(cells) != 7 or not cells[0].isdigit():
            continue
        index, parameter, *values = cells
        rows.append([garment, index, parameter, *values])

    garments = sorted({r[0] for r in rows})
    if len(rows) != 64 or garments != ["Blazer", "Long sleeved shirt",
                                       "Short sleeved shirt", "Trousers"]:
        print(f"unexpected shape: {len(rows)} rows, garments={garments}", file=sys.stderr)
        return 1

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["garment", "parameter_index", "parameter", "P1_cm", "P2_cm", "P3_cm", "P4_cm", "P5_cm"])
        w.writerows(rows)
    print(f"wrote {OUT.name}: {len(rows)} parameters across {len(garments)} garment types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
