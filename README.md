# QGimbal-vision

QGimbal云台配套视觉程序。试用 OpenCV 库实现基于传统视觉算法的白色矩形检测，将误差进行PID计算后，通过UART向QGimbal发送指令实现目标追踪。

## 运行步骤

1. **克隆仓库**

```bash
sudo apt install git # 若未安装git，请先安装
git clone https://github.com/Liu-Curiousity/QGimbal-Vision.git # 克隆仓库
cd QGimbal-Vision # 进入仓库目录
```

2. **安装依赖环境**

```bash
pip install -r requirements.txt # 使用pip安装
sudo apt install python3-opencv # 若pip安装opencv失败，可尝试apt安装
```

3. **运行程序**

选择其中一种模式运行：

- 窗口模式：显示摄像头图像，便于调试和观察识别效果，但帧率较低
- 无窗口模式：不显示摄像头图像，进通过终端输出基本信息，帧率高

**注意：** 
  1. 若使用树莓派40P引脚中的串口，需先使用`raspi-config`工具启用串口（其他XX派开启方式类似，不再赘述）。若使用USB转TTL模块，请根据实际情况修改串口号。
  2. 若使用SSH远程连接时选择窗口模式运行，请确保SSH客户端支持X11转发，并在连接时使用`ssh -X`参数。

```bash
# 窗口模式，使用串口ttyAMA0，按 `q` 或 `ESC` 退出
python main.py --serial-port /dev/ttyAMA0 --display 1
```

```bash
# 窗口模式，使用串口ttyAMA0，按 `Ctrl+C` 退出
python main.py --serial-port /dev/ttyAMA0 --display 0
```

## 说明

- 图像坐标：x 向右为正，y 向下为正
- 误差定义：`err = current_center - target_center`
- 若摄像头不在追踪目标处，可以更改`main.py`中的
  `tracker.target_center = (frame.shape[:2][1] // 2 + 15, frame.shape[:2][0] // 2)`调整`target_center`，使其与摄像头实际位置一致。
- 更多参数可使用`python main.py --help`查看