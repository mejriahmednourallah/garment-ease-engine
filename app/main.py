"""Garment ease engine: rank sizes by weighted squared ease and record fit feedback."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

REGIONS = ("chest", "waist", "hip")
WEIGHTS = {"chest": 0.4, "waist": 0.3, "hip": 0.3}
# Desired garment-minus-body gap in cm per fit preference.
TARGET_EASE = {
    "slim": {"chest": 2, "waist": 1, "hip": 2},
    "regular": {"chest": 8, "waist": 5, "hip": 8},
    "loose": {"chest": 14, "waist": 12, "hip": 14},
}
FIT_BAND_CM = 2.0
DB_ENV = "GARMENT_DB_PATH"


class Garment(BaseModel):
    size: str = Field(min_length=1, max_length=32)
    chest: float = Field(ge=30, le=300, allow_inf_nan=False)
    waist: float = Field(ge=30, le=300, allow_inf_nan=False)
    hip: float = Field(ge=30, le=300, allow_inf_nan=False)


class RecommendRequest(BaseModel):
    chest: float = Field(ge=30, le=250, allow_inf_nan=False)
    waist: float = Field(ge=30, le=250, allow_inf_nan=False)
    hip: float = Field(ge=30, le=250, allow_inf_nan=False)
    garments: list[Garment] = Field(min_length=1, max_length=50)
    fit_preference: Literal["slim", "regular", "loose"]
    stretch_percent: float = Field(ge=0, le=50, allow_inf_nan=False)

    @model_validator(mode="after")
    def unique_size_labels(self):
        labels = [garment.size for garment in self.garments]
        if any(not label.strip() or label != label.strip() for label in labels):
            raise ValueError("size labels must be nonblank without leading or trailing whitespace")
        if len(set(labels)) != len(labels):
            raise ValueError("size labels must be unique")
        return self


FitStatus = Literal["tight", "fits", "loose"]


class RegionFit(BaseModel):
    chest: FitStatus
    waist: FitStatus
    hip: FitStatus


class FeedbackRequest(BaseModel):
    request: RecommendRequest
    size: str = Field(min_length=1, max_length=32)
    actual_fit: RegionFit

    @model_validator(mode="after")
    def size_was_offered(self):
        if self.size not in {g.size for g in self.request.garments}:
            raise ValueError("size must be one of request.garments[].size")
        return self


def _status(delta: float) -> FitStatus:
    if delta <= -FIT_BAND_CM:
        return "tight"
    if delta >= FIT_BAND_CM:
        return "loose"
    return "fits"


def rank_sizes(body: dict, garments: list[dict], fit_preference: str, stretch_percent: float) -> list[dict]:
    """Ascending score = weighted squared gap between garment and body+target-stretch cm."""
    targets = TARGET_EASE[fit_preference]
    ranked = []
    for garment in garments:
        regions = {}
        score = 0.0
        for region in REGIONS:
            body_cm, garment_cm = body[region], garment[region]
            given = body_cm * stretch_percent / 100
            required = body_cm + targets[region] - given
            delta = garment_cm - required
            score += WEIGHTS[region] * delta * delta
            regions[region] = {
                "body_cm": body_cm,
                "garment_cm": garment_cm,
                "target_ease_cm": targets[region],
                "actual_ease_cm": round(garment_cm - body_cm, 2),
                "delta_cm": round(delta, 2),
                "status": _status(delta),
            }
        ranked.append({"size": garment["size"], "score": score, "regions": regions})
    ranked.sort(key=lambda item: (item["score"], item["size"]))
    for item in ranked:
        item["score"] = round(item["score"], 2)
    return ranked


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(os.environ.get(DB_ENV, "garment_ease.db"))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            size TEXT NOT NULL,
            fit_preference TEXT NOT NULL,
            stretch_percent REAL NOT NULL,
            request_json TEXT NOT NULL,
            actual_fit_json TEXT NOT NULL
        )"""
    )
    return conn


app = FastAPI(title="Garment Ease Engine")


@app.post("/recommend")
def recommend(payload: RecommendRequest):
    body = payload.model_dump()
    return {
        "fit_preference": payload.fit_preference,
        "stretch_percent": payload.stretch_percent,
        "weights": WEIGHTS,
        "ranked": rank_sizes(body, body["garments"], payload.fit_preference, payload.stretch_percent),
    }


@app.post("/feedback", status_code=201)
def feedback(payload: FeedbackRequest):
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn = connect()
    try:
        cur = conn.execute(
            "INSERT INTO feedback (created_at, size, fit_preference, stretch_percent, request_json, actual_fit_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                created_at,
                payload.size,
                payload.request.fit_preference,
                payload.request.stretch_percent,
                json.dumps(payload.request.model_dump()),
                json.dumps(payload.actual_fit.model_dump()),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": cur.lastrowid, "created_at": created_at}
