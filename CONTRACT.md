# Garment Ease Engine — API contract (v1)

Base path: same origin, JSON in/out. All lengths in **cm** (numbers, may be fractional).
No auth, no CORS config yet (add on the dashboard side / decide later).

## POST /recommend

Rank garment sizes for one body.

Request:

```json
{
  "chest": 96, "waist": 80, "hip": 102,
  "garments": [
    {"size": "S", "chest": 92, "waist": 76, "hip": 98},
    {"size": "M", "chest": 100, "waist": 84, "hip": 106}
  ],
  "fit_preference": "regular",
  "stretch_percent": 5
}
```

| field | type | rules |
|---|---|---|
| chest / waist / hip | number | body circumference, 30–250 |
| garments | array, 1–50 | `size` non-empty string; chest/waist/hip 30–300 |
| fit_preference | string | `slim` \| `regular` \| `loose` |
| stretch_percent | number | 0–50, percent of body circumference the fabric gives |

Response `200`:

```json
{
  "fit_preference": "regular",
  "stretch_percent": 5,
  "weights": {"chest": 0.4, "waist": 0.3, "hip": 0.3},
  "ranked": [
    {
      "size": "M",
      "score": 3.32,
      "regions": {
        "chest": {"body_cm": 96, "garment_cm": 100, "target_ease_cm": 8,
                  "actual_ease_cm": 4, "delta_cm": 0.8, "status": "fits"},
        "waist": {"body_cm": 80, "garment_cm": 84, "target_ease_cm": 5,
                  "actual_ease_cm": 4, "delta_cm": 3.0, "status": "loose"},
        "hip":   {"body_cm": 102, "garment_cm": 106, "target_ease_cm": 8,
                  "actual_ease_cm": 4, "delta_cm": 1.1, "status": "fits"}
      }
    }
  ]
}
```

`size` labels must be nonblank, unique, and have no surrounding whitespace.
`body_cm`, `garment_cm`, `actual_ease_cm` and `delta_cm` come back as JSON floats (96.0);
`target_ease_cm` is an int from the table below.
`ranked` is sorted by the unrounded score (lower = better), with exact ties broken by `size` string. Displayed scores are rounded to two decimals, so equal displayed scores may have different ranks.
`status` is per region: `tight` (delta <= -2), `fits` (|delta| < 2), `loose` (delta >= 2).

## Scoring (transparent, no ML)

Target ease by preference, cm:

| preference | chest | waist | hip |
|---|---|---|---|
| slim | 2 | 1 | 2 |
| regular | 8 | 5 | 8 |
| loose | 14 | 12 | 14 |

For each region: `required_cm = body_cm + target_ease_cm - body_cm * stretch_percent / 100`
`delta_cm = garment_cm - required_cm`; `actual_ease_cm = garment_cm - body_cm`.
`score = 0.4*delta_chest^2 + 0.3*delta_waist^2 + 0.3*delta_hip^2`, rounded to 2 dp.

## POST /feedback

Record how a recommended size actually fit. Stores the request verbatim plus the outcome.

```json
{
  "request": { ...exact POST /recommend body... },
  "size": "M",
  "actual_fit": {"chest": "tight", "waist": "fits", "hip": "loose"}
}
```

`size` must be one of `request.garments[].size`. `actual_fit` needs all three keys, each
`tight` | `fits` | `loose`.

Response `201`: `{"id": 1, "created_at": "2026-11-24T16:40:00+00:00"}`.

## Errors

`422` validation detail for any rule violation above; rejected input values are not echoed.
Unexpected server errors return HTTP 500. Nothing is logged for `/recommend`; only `/feedback` writes rows.

## Persistence

SQLite file path from env `GARMENT_DB_PATH` (default `./garment_ease.db`), table
`feedback(id, created_at, size, fit_preference, stretch_percent, request_json, actual_fit_json)`.
