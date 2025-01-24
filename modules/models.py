import json
import numpy as np
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, field_validator


class Coordinates(BaseModel):
    x: float
    y: float


class Shape(BaseModel):
    shape: List[Coordinates] = [Coordinates(x=0, y=0), Coordinates(
        x=0.5, y=0), Coordinates(x=0.5, y=1), Coordinates(x=0, y=1)]


class Xyxy(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x1", "y1", "x2", "y2", mode="before")
    def round_to_three_decimals(cls, value):
        return round(value, 3)


class YoloData(BaseModel):
    image: np.ndarray | List[np.ndarray]
    confidence: float
    classes: List[int]

    class Config:
        arbitrary_types_allowed = True


class CameraData(BaseModel):
    classes: List[int] = [0, 1, 2]
    confidence: float = 0.5
    masks: Optional[List[Shape]] = [Shape()]
    is_focus: bool = True


class Detection(BaseModel):
    bbox: Xyxy
    confidence: float
    class_id: int
    class_name: str


class Request(BaseModel):
    ip: str
    nvr_name: str
    channel_id: str
    event_time: datetime
    event_type: str
    snapshots: List[str]
    time_detect: datetime


class AlertsRequest(BaseModel):
    ip: str
    nvr_name: str
    channel_id: str
    event_time: datetime
    event_type: str
    snapshots: List[str]
    camera_data: CameraData = CameraData()

    def without_camera_data(self):
        # Create a dictionary without `camera_data` and add `time_detect`
        data = self.dict(exclude={"camera_data"})
        data["time_detect"] = str(datetime.now())  # Add current time
        return data


class AlertsResponse(BaseModel):
    camera_data: Request
    detections: List[List[Detection]] | List


class DetectionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)
