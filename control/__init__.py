"""Control modules (PID, gimbal tracking, serial I/O stubs).

This package converts vision detections (target center in image coordinates) into
control outputs (e.g., gimbal yaw/pitch speed in RPM).
"""

from .tracker_control import GimbalTracker
from .serial_stub import GimbalSerialStub
from .feedbacker import Feedbacker
from .pid import PID

__all__ = [
    "GimbalTracker",
    "GimbalSerialStub",
    "Feedbacker",
    "PID",
]
