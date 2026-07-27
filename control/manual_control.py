"""Keyboard manual gimbal control (WASD / arrow keys)."""

from __future__ import annotations

import time
from typing import Tuple


# Arrow key codes across common OpenCV HighGUI backends
_LEFT = frozenset({81, 65361, 2424832})
_UP = frozenset({82, 65362, 2490368})
_RIGHT = frozenset({83, 65363, 2555904})
_DOWN = frozenset({84, 65364, 2621440})


class ManualController:
    """Map keyboard input to yaw/pitch RPM commands.

    Key map (camera view):
      A / Left  → yaw -
      D / Right → yaw +
      W / Up    → pitch -
      S / Down  → pitch +
      Space     → stop immediately

    OpenCV only reports key-down (with OS key-repeat while held). After the
    last control key, RPM decays to zero once ``stop_timeout_s`` elapses.
    """

    def __init__(
        self,
        rpm: float = 10.0,
        stop_timeout_s: float = 0.15,
        invert_yaw: bool = False,
        invert_pitch: bool = False,
    ) -> None:
        self.rpm = abs(rpm)
        self.stop_timeout_s = stop_timeout_s
        self.invert_yaw = invert_yaw
        self.invert_pitch = invert_pitch
        self._yaw = 0.0
        self._pitch = 0.0
        self._last_cmd_ts = 0.0

    def reset(self) -> None:
        self._yaw = 0.0
        self._pitch = 0.0
        self._last_cmd_ts = 0.0

    def handle_key(self, key: int) -> None:
        if key is None or key < 0 or key == 255:
            return

        yaw, pitch = 0.0, 0.0
        matched = True
        if key in (ord("a"), ord("A")) or key in _LEFT:
            yaw = -self.rpm
        elif key in (ord("d"), ord("D")) or key in _RIGHT:
            yaw = self.rpm
        elif key in (ord("w"), ord("W")) or key in _UP:
            pitch = -self.rpm
        elif key in (ord("s"), ord("S")) or key in _DOWN:
            pitch = self.rpm
        elif key in (ord(" "),):
            yaw, pitch = 0.0, 0.0
        else:
            matched = False

        if not matched:
            return

        if self.invert_yaw:
            yaw = -yaw
        if self.invert_pitch:
            pitch = -pitch

        self._yaw = yaw
        self._pitch = pitch
        self._last_cmd_ts = time.time()

    def get_rpm(self) -> Tuple[float, float]:
        if self._yaw == 0.0 and self._pitch == 0.0:
            return 0.0, 0.0
        if (time.time() - self._last_cmd_ts) > self.stop_timeout_s:
            self._yaw = 0.0
            self._pitch = 0.0
            return 0.0, 0.0
        return self._yaw, self._pitch
