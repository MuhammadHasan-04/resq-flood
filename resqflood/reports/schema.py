"""Crowdsourced passability report model (matches DADU_DATA/09_Crowdsourced_Reports/report_schema.json)."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ReportType(str, Enum):
    road_open = "road_open"
    road_blocked = "road_blocked"
    bridge_damaged = "bridge_damaged"
    boat_needed = "boat_needed"
    water_rising = "water_rising"


class SourceRole(str, Enum):
    public = "public"
    ngo_worker = "ngo_worker"
    field_team = "field_team"
    authority = "authority"


class ReviewState(str, Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    conflict = "conflict"


class ReportIn(BaseModel):
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    timestamp_utc: datetime
    report_type: ReportType
    source_role: SourceRole
    note: str = Field(default="", max_length=280)
    photo_file: Optional[str] = None


class Report(ReportIn):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    osm_road_id: Optional[int] = None
    snap_distance_m: Optional[float] = None
    score: float = 0.0
    review_state: ReviewState = ReviewState.pending
    reasons: list[str] = []
