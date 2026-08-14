# 摄像头读取并显示画面
# 使用: python main.py --camera 0

"""
简单的摄像头预览脚本（使用 OpenCV）。
参数：
  --camera / --fps / --width / --height  采集参数（默认 640x480 @ 60fps）
  --display       是否显示图形化窗口（0/1，默认 1）
  --detect-mode   检测模式：rect / circle / object（默认 rect）
  --object-ref    物体检测的参考图像路径（detect-mode=object 时必需）
  --manual        启动时进入手动控制模式
  --manual-rpm    手动模式转速上限（RPM）
  --gamepad / --no-gamepad  启用/禁用 USB 手柄（默认启用并自动探测）

控制：
  GUI 模式按 'q' 或 ESC 退出；无窗口模式请按 Ctrl+C 退出。
  按 'm' 或手柄 Start 在自动追踪 / 手动控制之间切换。
  按 '1' / '2' / '3' 切换检测模式：矩形 / 圆形 / 物体。
  手动模式：
    - 手柄左摇杆 / 方向键：yaw / pitch（比例控制）
    - 手柄 A：急停；Back：退出
    - 键盘 WASD / 方向键 / 空格（需 --display 1）
  摄像头读帧连续失败时会自动 release 并重连。
"""

import argparse
import time
import sys

import cv2

from vision.detectors import create_detector, DetectionMode, BaseDetector

from control.pid import PID
from control.feedbacker import Feedbacker
from control.serial_stub import GimbalSerialStub
from control.tracker_control import GimbalTracker
from control.manual_control import ManualController
from control.gamepad_control import GamepadController

DEFAULT_CAMERA = 0  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 60  # 期望帧率（120 在 USB2 MJPEG 下易触发 Corrupt JPEG / 超时）
DEFAULT_DISPLAY = 1
DEFAULT_CAMERA_READ_FAILS = 3  # 连续读失败多少次后重开摄像头
DEFAULT_CAMERA_REOPEN_DELAY_S = 0.5

# 控制默认参数（可通过命令行覆盖）
DEFAULT_MAX_RPM = 20.0
DEFAULT_LOST_TIMEOUT_S = 0.4
DEFAULT_MANUAL_RPM = 1
DEFAULT_GAMEPAD_DEADZONE = 0.12

# 检测默认参数
DEFAULT_DETECT_MODE = "rect"


def parse_args():
    p = argparse.ArgumentParser(description="OpenCV 摄像头显示示例")
    p.add_argument('--camera', type=int, default=DEFAULT_CAMERA, help=f'摄像头索引（默认 {DEFAULT_CAMERA}）')
    p.add_argument('--display', type=int, choices=[0, 1], default=DEFAULT_DISPLAY,
                   help=f'是否显示图形化窗口（0/1，默认 {DEFAULT_DISPLAY}）')
    p.add_argument('--fps', type=float, default=DEFAULT_FPS,
                   help=f'摄像头期望帧率（默认 {DEFAULT_FPS}）')
    p.add_argument('--width', type=int, default=DEFAULT_WIDTH, help=f'期望宽度（默认 {DEFAULT_WIDTH}）')
    p.add_argument('--height', type=int, default=DEFAULT_HEIGHT, help=f'期望高度（默认 {DEFAULT_HEIGHT}）')

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

    # 检测相关
    p.add_argument('--detect-mode', type=str, choices=['rect', 'circle', 'object'],
                   default=DEFAULT_DETECT_MODE,
                   help=f'目标检测模式（默认 {DEFAULT_DETECT_MODE}）')
    p.add_argument('--object-ref', type=str, default=None,
                   help='物体检测的参考图像路径（detect-mode=object 时必需）')
    p.add_argument('--circle-min-circularity', type=float, default=0.7,
                   help='圆形检测最小圆度 0~1（默认 0.7）')
    p.add_argument('--circle-min-radius', type=int, default=10,
                   help='圆形检测最小半径（像素，默认 10）')
    p.add_argument('--circle-max-radius', type=int, default=300,
                   help='圆形检测最大半径（像素，默认 300）')
    p.add_argument('--object-min-matches', type=int, default=10,
                   help='ORB 匹配最小特征点数（默认 10）')

    return p.parse_args()


def _force_v4l2_fps(camera_index: int, fps: float) -> None:
    """Some UVC cams ignore OpenCV CAP_PROP_FPS; set via v4l2-ctl when available."""
    if not sys.platform.startswith("linux"):
        return
    device = f"/dev/video{camera_index}"
    try:
        import subprocess
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-parm={fps:g}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            msg = (result.stdout or result.stderr or "").strip()
            print(f"info: v4l2-ctl set-parm {fps:g} on {device}" + (f" ({msg})" if msg else ""))
        else:
            err = (result.stderr or result.stdout or "").strip()
            print(f"警告: v4l2-ctl 设置帧率失败: {err or result.returncode}")
    except FileNotFoundError:
        print("警告: 未安装 v4l2-ctl，无法强制摄像头帧率（可 apt install v4l-utils）")
    except Exception as exc:
        print(f"警告: 强制设置帧率异常: {exc}")


def open_camera(camera_index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    """Open and configure the capture device."""
    if sys.platform.startswith('linux'):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    elif sys.platform.startswith('win'):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return cap

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # 尽量只保留最新帧，减轻处理跟不上时的缓冲堆积
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _force_v4l2_fps(camera_index, fps)
    return cap


def reopen_camera(
    cap: cv2.VideoCapture | None,
    camera_index: int,
    width: int,
    height: int,
    fps: float,
    delay_s: float = DEFAULT_CAMERA_REOPEN_DELAY_S,
) -> cv2.VideoCapture | None:
    """Release and reopen the camera after a stream failure."""
    print(f"警告: 摄像头读帧失败，{delay_s:.1f}s 后尝试重连...")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    time.sleep(delay_s)
    new_cap = open_camera(camera_index, width, height, fps)
    if not new_cap.isOpened():
        print(f"警告: 摄像头重连失败 (index={camera_index})")
        try:
            new_cap.release()
        except Exception:
            pass
        return None
    print(
        f"info: 摄像头已重连 backend={new_cap.getBackendName()} "
        f"{new_cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} "
        f"@ {new_cap.get(cv2.CAP_PROP_FPS):.1f} fps"
    )
    return new_cap


def main():
    args = parse_args()

    cap = open_camera(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        print(f"无法打开摄像头索引 {args.camera}. 请检查设备或更换索引。")
        sys.exit(2)

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

    # 检测器初始化
    detector: BaseDetector | None = None
    detect_mode = args.detect_mode

    def set_detect_mode(new_mode: str) -> bool:
        nonlocal detector, detect_mode
        try:
            new_detector = create_detector(
                DetectionMode(new_mode),
                rect_min_area_ratio=0.005,
                rect_max_area_ratio=0.5,
                rect_angle_tol=25.0,
                circle_min_area_ratio=0.005,
                circle_max_area_ratio=0.5,
                circle_min_circularity=args.circle_min_circularity,
                circle_min_radius=args.circle_min_radius,
                circle_max_radius=args.circle_max_radius,
                object_ref_path=args.object_ref,
                object_min_matches=args.object_min_matches,
            )
            detector = new_detector
            detect_mode = new_mode
            print(f"切换到 {new_mode} 检测模式")
            return True
        except Exception as exc:
            print(f"警告: 无法切换到 {new_mode} 检测模式: {exc}")
            return False

    if not set_detect_mode(detect_mode):
        print("错误: 检测器初始化失败，退出")
        serial_stub.close()
        feedbacker.close()
        cap.release()
        sys.exit(2)

    print(f"info: control mode = {mode_name}  (键盘 m / 手柄 Start 切换)")
    print(f"info: detect mode = {detect_mode}  (键盘 1/2/3 切换)")
    print("info: 手动: 手柄左摇杆 或 WASD；A/空格急停；q/Back 退出")

    serial_stub.send_command(serial_stub.CmdType.EnableLaser)  # 启用激光
    serial_stub.send_command(serial_stub.CmdType.EnableStability)  # 启用陀螺仪稳定
    serial_stub.send_command(serial_stub.CmdType.Enable)  # 启用云台

    key = 255
    consecutive_read_fails = 0

    def toggle_manual_mode():
        nonlocal manual_mode
        manual_mode = not manual_mode
        tracker.enabled = not manual_mode
        tracker.reset()
        manual.reset()
        if gamepad is not None:
            gamepad.reset()
        if detector is not None:
            detector.reset()
        serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (0.0, 0.0))
        print(f"切换到 {'手动' if manual_mode else '自动追踪'} 模式")
        if manual_mode and not args.display and gamepad is None:
            print("警告: 当前无 GUI 且无手柄，无法接收控制输入")

    try:
        while True:
            if cap is None or not cap.isOpened():
                serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (0.0, 0.0))
                cap = reopen_camera(cap, args.camera, args.width, args.height, args.fps)
                consecutive_read_fails = 0
                if cap is None:
                    time.sleep(DEFAULT_CAMERA_REOPEN_DELAY_S)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_fails += 1
                print(f"无法从摄像头读取到帧 ({consecutive_read_fails}/{DEFAULT_CAMERA_READ_FAILS})...")
                serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (0.0, 0.0))
                if consecutive_read_fails >= DEFAULT_CAMERA_READ_FAILS:
                    cap = reopen_camera(cap, args.camera, args.width, args.height, args.fps)
                    consecutive_read_fails = 0
                else:
                    time.sleep(0.05)
                continue

            consecutive_read_fails = 0
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

            # 执行当前检测器（矩形 / 圆形 / 物体）
            targets = detector.detect(frame) if detector is not None else []
            best_target = targets[0] if targets else None
            best_target_center = best_target.center if best_target is not None else None

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
                error_pixel, yaw_pitch_rpm = tracker.update(frame.shape[:2], best_target_center)
                if yaw_pitch_rpm is not None:
                    yaw_rpm, pitch_rpm = yaw_pitch_rpm
                    serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (yaw_rpm, pitch_rpm))

            # 在画面上叠加检测框
            if best_target is not None:
                detector.draw(frame, best_target)

            mode_name = "manual" if manual_mode else "auto"
            key = feedbacker.update(
                frame, best_target, tracker.target_center, error_pixel, yaw_pitch_rpm,
                mode=mode_name, detect_mode=detect_mode,
            )

            if key in (ord('q'), ord('Q'), 27):  # q / ESC
                print("退出程序...")
                break
            if key in (ord('m'), ord('M')):
                toggle_manual_mode()
            if key == ord('1'):
                set_detect_mode('rect')
            if key == ord('2'):
                set_detect_mode('circle')
            if key == ord('3'):
                set_detect_mode('object')

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        if cap is not None:
            cap.release()
        if gamepad is not None:
            gamepad.close()
        serial_stub.close()
        feedbacker.close()


if __name__ == '__main__':
    main()
