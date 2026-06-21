"""
Curve model for poly dewarp — self-contained, no external model dependencies.
Ported from dewarp2/backend/models/curve_model.py
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field


class Point(BaseModel):
    x: float
    y: float


class Curve(BaseModel):
    id: str
    name: str
    color: str
    control_points: list[Point] = Field(min_length=7, max_length=7)
    sample_points: list[Point] = Field(default_factory=list)
    spline_points: list[Point] = Field(default_factory=list)
    spline_length: float = 0.0
    rmse: float = 0.0
