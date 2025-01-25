import json
import asyncio
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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


@dataclass
class MetricsTracker:
    start_time: datetime = field(default_factory=datetime.now)
    receives: int = 0
    sends: int = 0
    message_in_action: int = 0
    no_motion: int = 0
    no_detection: int = 0
    no_detection_on_mask: int = 0
    expires: int = 0
    errors: dict = field(default_factory=lambda: {
        'get': 0, 'send': 0, 'delete': 0, 'general': 0
    })

    processing_times: List[float] = field(default_factory=list)
    camera_to_detection_times: List[float] = field(default_factory=list)

    _metrics_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def update(self, metric_type: str, value: int = 1):
        async with self._metrics_lock:
            if hasattr(self, metric_type):
                current_value = getattr(self, metric_type)
                setattr(self, metric_type, current_value + value)

    async def add_processing_time(self, time: float, detection: bool = False):
        async with self._metrics_lock:
            self.processing_times.append(time)
            if detection:
                self.sends += 1

    async def add_camera_detection_time(self, time: float):
        async with self._metrics_lock:
            self.camera_to_detection_times.append(time)

    def calculate_metrics(self) -> dict:
        total_run_time = datetime.now() - self.start_time
        total_send_attempts = self.receives

        def format_time(time_delta: timedelta) -> str:
            units = [('Y', 365*24*60*60), ('M', 30*24*60*60), ('D', 24*60*60),
                     ('h', 3600), ('m', 60), ('s', 1)]
            total_seconds = int(time_delta.total_seconds())
            result = []
            for unit in units:
                unit_qty = total_seconds // unit[1]
                if unit_qty > 0:
                    result.append(f"{unit_qty}{unit[0]}")
                    total_seconds -= unit_qty * unit[1]
            return ' '.join(result) if result else '0s'

        def calculate_time_stats(times: List[float]) -> dict:
            return {
                'avg': format_time(timedelta(seconds=np.mean(times) if times else 0)),
                'median': format_time(timedelta(seconds=np.median(times) if times else 0)),
                'std': format_time(timedelta(seconds=np.std(times) if times else 0))
            }

        def calculate_rate(part: int, total: int) -> float:
            return round((part / total * 100) if total else 0, 3)

        total_errors = sum(self.errors.values())

        total_run_time_str = format_time(total_run_time)
        work_run_time_str = format_time(
            timedelta(seconds=sum(self.processing_times)))

        return {
            '🕒 total_run_time': total_run_time_str,
            '🔧 work_run_time': work_run_time_str,
            '📥 receives': self.receives,
            '📤 sends': self.sends,
            '🔄 message_in_action': self.message_in_action,
            '🚫🤖 no_motion': self.no_motion,
            '❌🔍 no_detection': self.no_detection,
            '🚫🎭 no_detection_on_mask': self.no_detection_on_mask,
            '⌛ expires': self.expires,
            '⚠️ errors': self.errors,
            '⏱️ camera_to_detection_times': calculate_time_stats(self.camera_to_detection_times),
            '📊 detection_rate': calculate_rate(self.sends, total_send_attempts),
            '⚠️📊 error_rate': calculate_rate(total_errors, total_send_attempts),
        }


metrics_tracker = MetricsTracker()
