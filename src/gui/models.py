from dataclasses import dataclass
from enum import Enum
from typing import List  # Add this import for Python 3.8

class RecordingState(Enum):
    STOPPED = "stopped"
    RECORDING = "recording"

@dataclass
class CameraSettings:
    resolution: str = "1920x1080"
    fps: int = 60
    debug_overlay: bool = False

@dataclass
class RaceModel:
    race_id: str
    version: str
    timestamp: str
    cars: List[str]  # Changed from list[str] to List[str]
    model_size: str
    status: str = "not_downloaded"  # not_downloaded, downloading, ready