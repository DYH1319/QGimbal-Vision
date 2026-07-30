# 摄像头读取并显示画面
# 使用: python main.py --camera 0

"""
简单的摄像头预览脚本（使用 OpenCV）。
参数：
  --camera        摄像头索引（默认 0）
  --display       是否显示图形化窗口（0/1，默认 1）
  --manual        启动时进入手动控制模式
  --manual-rpm    手动模式转速上限（RPM）
  --gamepad / --no-gamepad  启用/禁用 USB 手柄（默认启用并自动探测）

控制：
  GUI 模式按 'q' 或 ESC 退出；无窗口模式请按 Ctrl+C 退出。
  按 'm' 或手柄 Start 在自动追踪 / 手动控制之间切换。
  手动模式：
    - 手柄左摇杆 / 方向键：yaw / pitch（比例控制）
    - 手柄 A：急停；Back：退出
    - 键盘 WASD / 方向键 / 空格（需 --display 1）
"""

import argparse
import time
import sys

import cv2

from vision.rect_detect import detect_rectangles

from control.pid import PID
from control.feedbacker import Feedbacker
from control.serial_stub import GimbalSerialStub
from control.tracker_control import GimbalTracker
from control.manual_control import ManualController
from control.gamepad_control import GamepadController

DEFAULT_CAMERA = 0  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 120  # 期望帧率
DEFAULT_DISPLAY = 1

# 控制默认参数（可通过命令行覆盖）
DEFAULT_MAX_RPM = 20.0
DEFAULT_LOST_TIMEOUT_S = 0.4
DEFAULT_MANUAL_RPM = 1
DEFAULT_GAMEPAD_DEADZONE = 0.12


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--display', type=int, choices=[0, 1], default=DEFAULT_DISPLAY,
                   help=f'是否显示图形化窗口（0/1，默认 {DEFAULT_DISPLAY}）')

    # 控制相关
    p.add_argument('--max-rpm', type=float, default=DEFAULT_MAX_RPM,
                   help=f'最大转速输出（RPM，默认 {DEFAULT_MAX_RPM}）')
    p.add_argument('--lost-timeout', type=float, default=DEFAULT_LOST_TIMEOUT_S,
                   help=f'丢目标超时后复位控制器的时间（秒，默认 {DEFAULT_LOST_TIMEOUT_S}）')
    p.add_argument('--serial-port', type=str, default=None, help='串口端口号，例如 COM3；不填则不发送')
    p.add_argument('--serial-baud', type=int, default=115200, help='串口波特率（默认 115200）')
    p.add_argument('--manual', action='store_true', help='启动时进入手动控制模式')
    p.add_argument('--manual-rpm', type=float, default=DEFAULT_MANUAL_RPM,
                   help=f'手动模式转速（RPM，默认 {DEFAULT_MANUAL_RPM}）')
    p.add_argument('--gamepad', dest='gamepad', action='store_true', default=True,
                   help='启用 USB 手柄（默认开启）')
    p.add_argument('--no-gamepad', dest='gamepad', action='store_false',
                   help='禁用 USB 手柄')
    p.add_argument('--gamepad-index', type=int, default=0, help='手柄设备索引（默认 0）')
    p.add_argument('--gamepad-deadzone', type=float, default=DEFAULT_GAMEPAD_DEADZONE,
                   help=f'摇杆死区 0~1（默认 {DEFAULT_GAMEPAD_DEADZONE}）')

    return p.parse_args()


def main():
    args = parse_args()

    # 根据系统环境选择后端
    if sys.platform.startswith('linux'):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    elif sys.platform.startswith('win'):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"无法打开摄像头索引 {args.camera}. 请检查设备或更换索引。")
        sys.exit(2)

    # 设置参数
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 设置为 MJPG 格式
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
    print(f"info: capture backend: {cap.getBackendName()}")
    print(f"info: capture resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"info: capture FPS: {cap.get(cv2.CAP_PROP_FPS)}")

    # 控制器初始化
    tracker = GimbalTracker(
        yaw_pid=PID(kp=140.0, ki=40.0, kd=1.4, integral_limit=0.15, output_limit=args.max_rpm),
        pitch_pid=PID(kp=140.0, ki=40.0, kd=1.4, integral_limit=0.15, output_limit=args.max_rpm),
        lost_timeout_s=args.lost_timeout, invert_yaw=True
    )
    manual = ManualController(rpm=args.manual_rpm, invert_yaw=True)
    gamepad = None
    if args.gamepad:
        gamepad = GamepadController(
            index=args.gamepad_index,
            rpm=args.manual_rpm,
            deadzone=args.gamepad_deadzone,
            invert_yaw=True,
        )
        if not gamepad.connected:
            gamepad.close()
            gamepad = None

    serial_stub = GimbalSerialStub(args.serial_port, args.serial_baud)
    feedbacker = Feedbacker(display=args.display)
    serial_stub.open()

    manual_mode = bool(args.manual)
    if manual_mode and not args.display and gamepad is None:
        print("警告: 无手柄且 --display 0 时无法接收键盘，请接入手柄或使用 --display 1")

    tracker.enabled = not manual_mode
    mode_name = "manual" if manual_mode else "auto"
    print(f"info: control mode = {mode_name}  (键盘 m / 手柄 Start 切换)")
    print("info: 手动: 手柄左摇杆 或 WASD；A/空格急停；q/Back 退出")

    serial_stub.send_command(serial_stub.CmdType.EnableLaser)  # 启用激光
    serial_stub.send_command(serial_stub.CmdType.EnableStability)  # 启用陀螺仪稳定
    serial_stub.send_command(serial_stub.CmdType.Enable)  # 启用云台

    key = 255

    def toggle_manual_mode():
        nonlocal manual_mode
        manual_mode = not manual_mode
        tracker.enabled = not manual_mode
        tracker.reset()
        manual.reset()
        if gamepad is not None:
            gamepad.reset()
        serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (0.0, 0.0))
        print(f"切换到 {'手动' if manual_mode else '自动追踪'} 模式")
        if manual_mode and not args.display and gamepad is None:
            print("警告: 当前无 GUI 且无手柄，无法接收控制输入")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法从摄像头读取到帧，正在重试...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, -1)  # 翻转画面

            gp_events = gamepad.update() if gamepad is not None else None
            if gp_events is not None:
                if gp_events.quit:
                    print("手柄 Back：退出程序...")
                    break
                if gp_events.mode_toggle:
                    toggle_manual_mode()
                if gp_events.stop and manual_mode:
                    manual.reset()
                    serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (0.0, 0.0))

            # 对每帧执行矩形检测（自动模式用于追踪，手动模式仅用于叠加显示）
            rects = detect_rectangles(frame, min_area_ratio=0.005, max_area_ratio=0.5, angle_tol=25.0)
            best_rect = rects[0] if rects else None
            best_rect_center = rects[0].center if rects else None

            tracker.target_center = (frame.shape[:2][1] // 2 + 10, frame.shape[:2][0] // 2)

            if manual_mode:
                manual.handle_key(key)
                key_rpm = manual.get_rpm()
                if gamepad is not None and (gamepad.stick_active() or not any(key_rpm)):
                    yaw_rpm, pitch_rpm = gamepad.get_rpm()
                else:
                    yaw_rpm, pitch_rpm = key_rpm
                yaw_pitch_rpm = (yaw_rpm, pitch_rpm)
                error_pixel = (0.0, 0.0)
                serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, yaw_pitch_rpm)
            else:
                # PID 控制：将目标中心追踪到屏幕中心，输出 yaw/pitch rpm
                error_pixel, yaw_pitch_rpm = tracker.update(frame.shape[:2], best_rect_center)
                if yaw_pitch_rpm is not None:
                    yaw_rpm, pitch_rpm = yaw_pitch_rpm
                    serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (yaw_rpm, pitch_rpm))

            mode_name = "manual" if manual_mode else "auto"
            key = feedbacker.update(
                frame, best_rect, tracker.target_center, error_pixel, yaw_pitch_rpm, mode=mode_name
            )

            if key in (ord('q'), ord('Q'), 27):  # q / ESC
                print("退出程序...")
                break
            if key in (ord('m'), ord('M')):
                toggle_manual_mode()

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        if gamepad is not None:
            gamepad.close()
        serial_stub.close()
        feedbacker.close()


if __name__ == '__main__':
    main()
