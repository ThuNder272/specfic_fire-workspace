# 项目使用说明
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
conda activate robomaster


python aim_scheduler.py --port /dev/ttyTHS1 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --pnp-profile mer_131_6mm --no-show-tx  --target-color red



'''
  sudo systemctl daemon-reload
  sudo systemctl restart specific_fire.service
'''

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活Python环境
conda activate py310

# 检查依赖
pip list | grep -E "(torch|opencv|numpy|matplotlib)"
```

### 2. 核心功能演示

#### A. 实时锁定 + 串口发送（主流程）
```bash
source /home/nvidia/miniforge3/etc/profile.d/conda.sh
python aim_scheduler.py --port /dev/ttyTHS1 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --pnp-profile mer_131_6mm --no-show-tx --target-color blue

# 保留蓝色，但排除蓝2（rm4pt class id 2）
python aim_scheduler.py --port /dev/ttyTHS1 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --pnp-profile mer_131_6mm --no-show-tx --target-color blue --exclude-class-ids 2

# 同时只保留蓝2和红2（rm4pt class id 2 和 11，白名单模式）
python aim_scheduler.py --port /dev/ttyTHS1 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --pnp-profile mer_131_6mm --no-show-tx --target-class-ids 2,11


python aim_scheduler.py --port /dev/ttyTHS1 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --daheng-config "/home/nvidia/Desktop/specific_fire/MER-131-210U3C(KE0220070351).txt" --pnp-profile mer_131_6mm --no-show-tx --target-color blue




sudo chmod 666 /dev/ttyTHS1


python aim_scheduler.py --port /dev/ttyUSB0 --baud 115200 --rate 50 --show-window
python aim_scheduler.py --port /dev/ttyUSB0 --baud 115200 --rate 50 --show-window --gun-offset-y 79 --model-bullet-speed 0

python aim_scheduler.py --port /dev/ttyUSB0 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --ballistic-enable --drag-k 0.02 --pitch-min -10 --pitch-max 30 --ballistic-dt-ms 1.0 --yolo-max-det 1 --yolo-imgsz 640 --pred-async --no-show-tx --yolo-log-speed


python aim_scheduler.py --gun-offset-y 42 --no-show-tx --target-color blue --port /dev/ttyTHS1 --baud 115200

```

  source /home/nvidia/miniforge3/etc/profile.d/conda.sh
  conda activate robomaster

  python aim_scheduler.py --port /dev/ttyTHS1 --baud 115200 --rate 210 --show-window --gun-offset-y 42 --daheng-config "/home/nvidia/Desktop/specific_fire/MER-131-210U3C(KE0220070351).txt" --pnp-profile mer_131_6mm --no-show-tx  --target-color red

**功能**:
- YOLO检测装甲板
- 预测目标中心
- PnP解算角度
- 每周期发送一帧（预测）
> 默认打印串口发送数据；如需关闭，加 `--no-show-tx`。
> 默认使用大恒相机，并加载 `/workspace/RobotMaster/paper/MER-139-210U3C(KE0210010001).txt`。  
> 如需使用电脑摄像头，加 `--use-opencv`。  
> 如需指定其他相机配置文件，可加：`--daheng-config <path>`。  
> 若大恒相机不可用，当前不会自动回退到电脑摄像头。
> 默认开启显示窗口与 yaw 反向；如需关闭，用 `--no-show-window` / `--no-invert-yaw`。
> 若转动方向相反，可加 `--invert-yaw` 或 `--invert-pitch` 进行修正。
> 可用 `--max-yaw-rate`/`--max-pitch-rate` 限制角速度。
> 可用 `--lost-threshold` 设置连续丢失多少次后进入扫描模式（默认 30）。
> 开火门控参数：`--fire-confidence-threshold`（高于阈值开火）与 `--fire-force-interval-ms`（低置信度时保底间隔，默认 1000ms；仅在有预测结果时生效）。
> 弹道补偿参数：`--ballistic-enable` / `--drag-k` / `--pitch-min` / `--pitch-max` / `--ballistic-dt-ms`，可选 `--use-ballistic-time`。

**可选显示窗口**:
```bash
python aim_scheduler.py --port /dev/ttyUSB0 --baud 115200 --rate 50 --show-window
```

#### B. 固定串口发送（调试）
```bash
python camera_adaptation/uart_sender.py --port /dev/ttyUSB0 --baud 115200 --rate 50
```

#### C. 坐标预测演示（已归档）
```bash
python archive/demos/coordinate_prediction_demo.py
```

#### D. 多模型性能对比（已归档）
```bash
python archive/analysis/coordinate_prediction_comparison.py
```

## 📊 性能对比结果

### 第15帧预测性能排名

| 排名 | 模型 | MSE | MAE | PSNR | 推理时间 |
|------|------|-----|-----|-------|----------|
| **🥇 1** | **LSTM-KF-Fusion-Hierarchical** | **0.000010 ± 0.000090** | **0.000064 ± 0.000400** | **93.46 ± 7.58** | **3.11 ± 1.31 ms** |
| 🥈 2 | Standalone-LSTM | 0.000011 ± 0.000089 | 0.000908 ± 0.000388 | 60.46 ± 3.51 | 0.65 ± 0.06 ms |
| 🥉 3 | Standalone-Kalman | 0.000026 ± 0.000087 | 0.003999 ± 0.000458 | 47.89 ± 3.85 | 19.43 ± 0.28 ms |

## 🎯 关键特性

### 1. 动态中心点标记
- 中心点根据预测结果动态变化
- 不再固定在图像中心
- 基于预测帧的亮度分析

### 2. 三列视频布局
- **左侧**: 原始视频帧
- **中间**: YOLO检测结果（绿色框 + 标签）
- **右侧**: 第15帧预测（红色中心点标记）

### 3. 高质量视频输出
- 支持多种编码器
- 保持原始视频质量
- 智能编码器选择

## 🔧 参数配置

### 主要参数
```python
# 在 aim_scheduler.py 中通过命令行参数配置
python aim_scheduler.py --help
```

### 模型参数
```python
# 在 archive/analysis/coordinate_prediction_comparison.py 中修改
num_test_sequences = 200                    # 测试序列数量
sequence_length = 14                        # 输入序列长度
```

### 训练策略（PnP步长）
- 训练脚本 `train_coordinate_prediction.py` 使用PnP估计预测步长，失败回退默认步长。
- 可在脚本中配置 `input_sequence_length`、`max_lead_frames`、`bullet_speed_mps`、`system_latency_s`、`pnp_profile` 等参数。

## 📁 文件说明

### 核心模型文件
- `lstm_kf_fusion_model.py`: 融合模型主文件
- `standalone_lstm_model.py`: 独立LSTM模型
- `standalone_kalman_model.py`: 独立Kalman Filter模型
- `fusion_module.py`: 融合策略模块

### 数据处理文件
- `data_processor.py`: 数据处理器（YOLO集成）
- `lstm_module.py`: LSTM特征提取器
- `kalman_filter.py`: Kalman Filter模块

### 演示和评估文件（已归档）
- `archive/demos/coordinate_prediction_demo.py`: 坐标预测演示
- `archive/analysis/coordinate_prediction_comparison.py`: 多模型坐标预测对比

### 预训练模型
- `fusion_strategy_models/`: 融合模型权重
- `detection_models/`: 独立模型权重

## 🎮 使用示例

### 1. 自定义模型对比
```python
from archive.analysis.coordinate_prediction_comparison import CoordinatePredictionComparison

evaluator = CoordinatePredictionComparison()
evaluator.load_models()

# 创建自定义数据集
dataset = evaluator.create_coordinate_dataset(
    video_path="your_video.mp4",
    num_test_sequences=500,    # 增加测试序列数量
    confidence_threshold=0.4,
    context_padding=20
)

# 评估性能
results = evaluator.evaluate_coordinate_prediction(dataset)
```

## 🔍 故障排除

### 1. YOLO模型加载失败
```bash
# 检查模型文件
ls -la best.pt

# 检查网络连接
ping github.com

# 使用本地缓存
export TORCH_HOME=/home/krisy/.cache/torch
```

### 2. CUDA内存不足
```python
# 减少测试序列数量
num_test_sequences = 100  # 从200减少到100

# 减少序列长度
sequence_length = 10      # 从14减少到10
```

### 3. 视频编码器问题
```python
# 尝试不同的编码器
encoders = ['mp4v', 'avc1', 'XVID']

# 降低视频质量
high_quality = False
```

## 📈 性能优化建议

### 1. 硬件优化
- 使用CUDA支持的GPU
- 确保足够的GPU内存（建议8GB+）
- 使用SSD存储提高I/O性能

### 2. 参数调优
- 根据视频内容调整`confidence_threshold`
- 根据物体大小调整`context_padding`
- 根据需求平衡质量和速度

### 3. 批量处理
- 对于多个视频，可以批量处理
- 使用多进程提高效率
- 合理设置内存使用

## 🎯 学术应用

### 1. 论文写作
- 完整的性能对比数据
- 标准化的评估指标
- 可重现的实验结果

### 2. 实验设计
- 支持自定义数据集
- 灵活的模型配置
- 详细的性能分析

### 3. 结果展示
- 高质量的可视化图表
- 清晰的性能对比
- 专业的演示视频

---

**注意**: 首次运行可能需要下载YOLO模型，请确保网络连接正常。
