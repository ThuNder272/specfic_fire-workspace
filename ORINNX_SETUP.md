# Orin NX `robomaster` 环境说明

## 当前目标

这份说明对应当前机器：

- JetPack 6.0 / L4T 36.3.0
- CUDA 12.2
- Python 3.10
- conda 环境名：`robomaster`

这套环境已经在本机验证通过：

- `torch 2.4.0a0+3bcc3cddb5.nv24.07`
- `torchvision 0.19.0a0+48b1edf`（源码编译）
- `ultralytics 8.3.240`
- `cv2 4.8.0`
- `tensorrt 8.6.2`

## 激活方式

本机没有把 conda 写进 shell 初始化文件，手动激活：

```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate robomaster
```

激活后会自动生效两件事：

- 复用系统 Python 绑定：
  - `/usr/lib/python3.10/dist-packages`
  - `/usr/local/lib/python3.10/dist-packages`
- 设置 Jetson 运行时变量：
  - `LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cusparselt/lib:...`
  - `PYTHONNOUSERSITE=1`

## 复现步骤

1. 安装 Miniforge 到 `/home/nvidia/miniforge3`
2. 创建环境：

```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda create -y -n robomaster python=3.10 pip
conda activate robomaster
```

3. 在环境里加入系统 `.pth` 路径：

```bash
printf '%s\n%s\n' \
  '/usr/lib/python3.10/dist-packages' \
  '/usr/local/lib/python3.10/dist-packages' \
  > /home/nvidia/miniforge3/envs/robomaster/lib/python3.10/site-packages/system_jetson_dist_packages.pth
```

4. 安装 [requirements.orinnx.txt](/home/nvidia/specific_fire/requirements.orinnx.txt) 里的基础包
5. 安装 NVIDIA Jetson PyTorch wheel
6. 从 `vision` 源码编译 `torchvision`

## 验证命令

```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate robomaster
python - <<'PY'
import torch, torchvision, ultralytics, cv2, tensorrt, serial
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
print(torchvision.__version__)
print(ultralytics.__version__)
print(cv2.__version__)
print(tensorrt.__version__)
print(serial.__version__)
PY
```

```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate robomaster
python aim_scheduler.py --help
```

```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate robomaster
python - <<'PY'
from ultralytics import YOLO
YOLO('best.pt', task='detect')
YOLO('best.engine', task='detect')
print('detector load ok')
PY
```

## 运行提示

- 当前代码默认走大恒相机；如果没有大恒相机，可手动加 `--use-opencv`
- 当前机器上还没有检测到串口设备和相机设备，所以完整主流程运行前需要把硬件接上
- 主流程最小启动示例：

```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate robomaster
python aim_scheduler.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --rate 50 \
  --gun-offset-y 42 \
  --no-show-window \
  --no-show-tx
```
