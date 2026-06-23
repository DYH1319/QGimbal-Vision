from __future__ import annotations

import time
from typing import Tuple
from .pid import PID


class GimbalTracker:
    """Convert image target center to gimbal yaw/pitch RPM commands."""

    def __init__(self,
                 yaw_pid: PID,
                 pitch_pid: PID,
                 lost_timeout_s: float,
                 invert_yaw: bool = False,
                 invert_pitch: bool = False) -> None:
        self.enabled = False
        self.target_center = (0.0, 0.0)
        self._lost_timeout_s = lost_timeout_s
        self._invert_yaw = invert_yaw
        self._invert_pitch = invert_pitch
        self._yaw_pid = yaw_pid
        self._pitch_pid = pitch_pid
        self._last_seen_ts = 0.0
        self._prev_time = 0.0

    def reset(self) -> None:
        self._yaw_pid.reset()
        self._pitch_pid.reset()
        self._prev_time = None
        self._last_seen_ts = 0.0

    def update(
            self,
            frame_shape: Tuple[float, float],
            current_center: Tuple[float, float] | None
    ) -> tuple[tuple[float, float], tuple[float, float] | None]:
        """Compute RPM commands.

        Args:
            frame_shape: image shape.
            current_center: (cx, cy) in image pixels, or None when target not found.
        """
        now = time.time()
        if not self.enabled or frame_shape is None:
            return (0, 0), (0, 0)

        # Lost target handling
        if current_center is None:
            if self._last_seen_ts > 0 and (now - self._last_seen_ts) <= self._lost_timeout_s:
                # within grace period: keep trying with zero error (hold still).
                return (0, 0), None
            self.reset()
            return (0, 0), (0, 0)
        self._last_seen_ts = now

        err_x = current_center[0] - self.target_center[0]
        err_y = current_center[1] - self.target_center[1]

        err_xu = err_x / frame_shape[1]  # normalize to [-0.5, 0.5]
        err_yu = err_y / frame_shape[0]  # normalize to [-0.5, 0.5]

        yaw_rpm = self._yaw_pid.update(err_xu, self._prev_time - now)
        pitch_rpm = self._pitch_pid.update(err_yu, self._prev_time - now)

        if self._invert_yaw: yaw_rpm = -yaw_rpm
        if self._invert_pitch: pitch_rpm = -pitch_rpm

        self._prev_time = now
        return (err_x, err_y), (yaw_rpm, pitch_rpm)
