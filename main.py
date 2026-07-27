# 摄像头读取并显示画面
# 使用: python main.py --camera 0

"""
简单的摄像头预览脚本（使用 OpenCV）。
参数：
  --camera        摄像头索引（默认 0）
  --display       是否显示图形化窗口（0/1，默认 1）
                 - 1：显示图像（现有效果），并叠加矩形与 FPS
                 - 0：不显示窗口，在终端输出 FPS + 检测到的矩形中心点坐标/面积
  --print-interval 终端输出间隔秒数（仅 --display 0 时生效，默认 0.5）
  --manual        启动时进入键盘手动控制模式
  --manual-rpm    手动模式转速（RPM）

控制：
  GUI 模式按 'q' 或 ESC 退出；无窗口模式请按 Ctrl+C 退出。
  按 'm' 在自动追踪 / 手动控制之间切换（需 --display 1）。
  手动模式：WASD 或方向键控制 yaw/pitch，空格急停。
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

DEFAULT_CAMERA = 0  # 摄像头索引（默认 0）
DEFAULT_WIDTH = 640  # 期望宽度
DEFAULT_HEIGHT = 480  # 期望高度
DEFAULT_FPS = 120  # 期望帧率
DEFAULT_DISPLAY = 1

# 控制默认参数（可通过命令行覆盖）
DEFAULT_MAX_RPM = 20.0
DEFAULT_LOST_TIMEOUT_S = 0.4
DEFAULT_MANUAL_RPM = 1


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
    p.add_argument('--manual', action='store_true', help='启动时进入键盘手动控制模式')
    p.add_argument('--manual-rpm', type=float, default=DEFAULT_MANUAL_RPM,
                   help=f'手动模式转速（RPM，默认 {DEFAULT_MANUAL_RPM}）')

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
    serial_stub = GimbalSerialStub(args.serial_port, args.serial_baud)
    feedbacker = Feedbacker(display=args.display)
    serial_stub.open()

    manual_mode = bool(args.manual)
    if manual_mode and not args.display:
        print("警告: 键盘手动控制需要 --display 1（OpenCV 窗口才能接收按键）")

    tracker.enabled = not manual_mode
    mode_name = "manual" if manual_mode else "auto"
    print(f"info: control mode = {mode_name}  (按 m 切换，需 GUI)")
    print("info: 手动键位 WASD/方向键，空格急停，q 退出")

    serial_stub.send_command(serial_stub.CmdType.EnableLaser)  # 启用激光
    serial_stub.send_command(serial_stub.CmdType.EnableStability)  # 启用陀螺仪稳定
    serial_stub.send_command(serial_stub.CmdType.Enable)  # 启用云台

    key = 255
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("无法从摄像头读取到帧，正在重试...")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, -1)  # 翻转画面

            # 对每帧执行矩形检测（自动模式用于追踪，手动模式仅用于叠加显示）
            rects = detect_rectangles(frame, min_area_ratio=0.005, max_area_ratio=0.5, angle_tol=25.0)
            best_rect = rects[0] if rects else None
            best_rect_center = rects[0].center if rects else None

            tracker.target_center = (frame.shape[:2][1] // 2 + 10, frame.shape[:2][0] // 2)

            if manual_mode:
                manual.handle_key(key)
                yaw_rpm, pitch_rpm = manual.get_rpm()
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
                manual_mode = not manual_mode
                tracker.enabled = not manual_mode
                tracker.reset()
                manual.reset()
                serial_stub.send_command(serial_stub.CmdType.LowSpeedCtrl, (0.0, 0.0))
                print(f"切换到 {'手动' if manual_mode else '自动追踪'} 模式")
                if manual_mode and not args.display:
                    print("警告: 当前无 GUI，无法接收键盘；请使用 --display 1")

    except KeyboardInterrupt:
        print('\n收到中断，退出...')
    finally:
        cap.release()
        serial_stub.close()
        feedbacker.close()


if __name__ == '__main__':
    main()
