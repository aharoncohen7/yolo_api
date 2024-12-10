import numpy as np
from datetime import datetime
from pydantic import BaseModel, field_validator
from typing import Optional, List


class Coordinates(BaseModel):
    x: float
    y: float


class Shape(BaseModel):
    shape: List[Coordinates]


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
    confidence: float = 0.2
    masks: Optional[List[Shape]] = []
    is_focus: bool = True


class Detection(BaseModel):
    bbox: Xyxy
    confidence: float
    class_id: int
    class_name: str


class AlertRequest(BaseModel):
    url: str
    camera_data: CameraData = CameraData()


class AlertsRequest(BaseModel):
    urls: List[str]
    camera_data: CameraData = CameraData()


class AlertResponse(BaseModel):
    url: str
    camera_data: CameraData
    detections: List[Detection] | List
    time_created: datetime = datetime.now()


class AlertsResponse(BaseModel):
    urls: List[str]
    camera_data: CameraData
    detections: List[List[Detection]] | List
    time_detect: datetime = datetime.now()
