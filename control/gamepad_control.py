"""USB gamepad control for gimbal (Yahboom ROS / Xbox-mode HID)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

# OpenCV owns the window; keep pygame headless for joystick only.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame


@dataclass
class GamepadEvents:
    mode_toggle: bool = False
    stop: bool = False
    quit: bool = False


class GamepadController:
    """Read a Linux joystick and map left stick to yaw/pitch RPM.

    Default layout matches Yahboom ROS USB pad in X-BOX mode (and most
    Xbox-compatible HID pads under pygame/SDL2):

      axis 0  left stick X  → yaw
      axis 1  left stick Y  → pitch
      button 0 (A)          → emergency stop
      button 7 (Start)      → auto / manual toggle
      button 6 (Back)       → quit request (optional)

    Triggers often sit at -1.0 when idle; they are ignored.
    """

    def __init__(
        self,
        index: int = 0,
        rpm: float = 10.0,
        deadzone: float = 0.12,
        invert_yaw: bool = False,
        invert_pitch: bool = False,
        yaw_axis: int = 0,
        pitch_axis: int = 1,
        stop_button: int = 0,
        mode_button: int = 7,
        quit_button: int = 6,
    ) -> None:
        self.index = index
        self.rpm = abs(rpm)
        self.deadzone = max(0.0, min(deadzone, 0.9))
        self.invert_yaw = invert_yaw
        self.invert_pitch = invert_pitch
        self.yaw_axis = yaw_axis
        self.pitch_axis = pitch_axis
        self.stop_button = stop_button
        self.mode_button = mode_button
        self.quit_button = quit_button

        self._js: pygame.joystick.Joystick | None = None
        self._prev_buttons: list[int] = []
        self._yaw = 0.0
        self._pitch = 0.0
        self._force_stop = False

        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        self.open()

    @property
    def connected(self) -> bool:
        return self._js is not None

    def open(self) -> bool:
        self.close()
        count = pygame.joystick.get_count()
        if count <= 0 or self.index >= count:
            print(f"info: 未找到手柄 (index={self.index}, count={count})")
            return False
        js = pygame.joystick.Joystick(self.index)
        js.init()
        self._js = js
        self._prev_buttons = [0] * js.get_numbuttons()
        print(
            f"info: 手柄已连接 name={js.get_name()!r} "
            f"axes={js.get_numaxes()} buttons={js.get_numbuttons()} hats={js.get_numhats()}"
        )
        print(
            "info: 手柄映射 左摇杆=yaw/pitch  A=急停  Start=切换模式  Back=退出"
        )
        return True

    def close(self) -> None:
        if self._js is not None:
            try:
                self._js.quit()
            except Exception:
                pass
            self._js = None

    def reset(self) -> None:
        self._yaw = 0.0
        self._pitch = 0.0
        self._force_stop = False

    def _apply_deadzone(self, value: float) -> float:
        if abs(value) < self.deadzone:
            return 0.0
        # Rescale so motion starts smoothly after deadzone
        sign = 1.0 if value > 0 else -1.0
        scaled = (abs(value) - self.deadzone) / (1.0 - self.deadzone)
        return sign * min(scaled, 1.0)

    def update(self) -> GamepadEvents:
        """Pump events and refresh stick/button state. Call once per frame."""
        events = GamepadEvents()
        if self._js is None:
            return events

        pygame.event.pump()
        js = self._js

        # Re-open if device vanished
        try:
            _ = js.get_axis(0)
        except pygame.error:
            print("警告: 手柄断开，尝试重连...")
            self.open()
            return events

        n_axes = js.get_numaxes()
        ax = js.get_axis(self.yaw_axis) if self.yaw_axis < n_axes else 0.0
        ay = js.get_axis(self.pitch_axis) if self.pitch_axis < n_axes else 0.0
        ax = self._apply_deadzone(ax)
        ay = self._apply_deadzone(ay)

        yaw = ax * self.rpm
        pitch = ay * self.rpm
        if self.invert_yaw:
            yaw = -yaw
        if self.invert_pitch:
            pitch = -pitch

        # D-pad as digital override when stick is neutral
        if js.get_numhats() > 0 and ax == 0.0 and ay == 0.0:
            hx, hy = js.get_hat(0)
            if hx or hy:
                yaw = float(hx) * self.rpm
                pitch = float(-hy) * self.rpm  # hat Y: up = +1
                if self.invert_yaw:
                    yaw = -yaw
                if self.invert_pitch:
                    pitch = -pitch

        n_btn = js.get_numbuttons()
        buttons = [js.get_button(i) for i in range(n_btn)]
        prev = self._prev_buttons
        if len(prev) != n_btn:
            prev = [0] * n_btn

        def pressed(i: int) -> bool:
            return i < n_btn and buttons[i] and not prev[i]

        if pressed(self.stop_button):
            events.stop = True
            self._force_stop = True
            yaw, pitch = 0.0, 0.0
        if pressed(self.mode_button):
            events.mode_toggle = True
        if pressed(self.quit_button):
            events.quit = True

        # Clear force-stop once stick leaves deadzone again
        if self._force_stop:
            if abs(ax) > 0.0 or abs(ay) > 0.0:
                self._force_stop = False
            else:
                yaw, pitch = 0.0, 0.0

        self._yaw = yaw
        self._pitch = pitch
        self._prev_buttons = buttons
        return events

    def get_rpm(self) -> Tuple[float, float]:
        return self._yaw, self._pitch

    def stick_active(self) -> bool:
        return abs(self._yaw) > 1e-6 or abs(self._pitch) > 1e-6
