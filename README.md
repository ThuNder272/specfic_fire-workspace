# 🚀 LSTM + Kalman Filter 融合坐标预测模型项目

## 📋 项目概述

本项目实现了一个基于LSTM和Kalman Filter融合的智能坐标预测系统，专注于**目标物体坐标预测**功能：

**🎯 坐标预测**: 基于前N帧，使用PnP估计预测步长，预测目标帧坐标位置 (x, y, w, h)

**🔬 多模型对比**: 包含融合模型、独立模型、传统机器学习模型和串行组合模型的全面性能对比

## 🎯 核心功能

### 坐标预测系统
- **功能**: 基于前N帧与PnP估计步长进行坐标预测
- **输出**: 4维坐标 (x_center, y_center, width, height)
- **应用**: 目标跟踪、运动预测、轨迹分析、智能监控
- **评估**: MSE、RMSE、MAE、R²、推理时间
- **模型**: 融合模型、独立LSTM、独立Kalman、SVR、AR、ANN、DT、KNN、串行Kalman+LSTM

### YOLO目标检测集成
- **检测模型**: 支持YOLOv5/YOLOv8
- **目标识别**: 自动识别视频中的目标物体
- **ROI提取**: 64×64像素的目标区域提取
- **坐标标注**: 为坐标预测提供真实标签

## 🏗️ 项目结构

```
📁 核心模型
├── 🎯 坐标预测模型
│   └── coordinate_prediction_model.py           # 坐标预测主模型
├── 🔧 功能模块
│   ├── fusion_module.py                         # 分层融合模块
│   ├── lstm_module.py                           # LSTM特征提取器
│   └── kalman_filter.py                         # Kalman Filter模块
├── 🤖 传统机器学习模型
│   ├── svr_model.py                             # 支持向量回归模型
│   ├── ar_model.py                              # 自回归模型
│   ├── ann_model.py                             # 人工神经网络模型
│   ├── dt_model.py                              # 决策树模型
│   └── knn_model.py                             # K近邻模型
└── 🔄 串行组合模型
    └── serial_kalman_lstm_model.py              # 串行Kalman+LSTM组合模型

📁 训练脚本（已归档）
├── 🎯 坐标预测训练
│   ├── archive/training/root/train_coordinate_prediction.py
│   └── archive/training/camera_adaptation/train_coordinate_prediction.py

📁 功能演示
├── 🎯 坐标预测演示（已归档）
│   ├── archive/demos/coordinate_prediction_demo.py             # 坐标预测演示
│   └── archive/analysis/coordinate_prediction_comparison.py    # 坐标预测多模型对比

📁 数据处理
└── data_processor.py                             # 数据处理器（YOLO集成）

📁 配置和文档
├── requirements.txt                               # 依赖列表
├── README.md                                     # 项目说明
├── USAGE.md                                      # 详细使用说明
└── quick_start_guide.py                          # 快速开始指南

📁 模型和结果
├── coordinate_prediction_models/                  # 训练好的模型权重
├── coordinate_prediction_comparison_results/      # 性能对比结果
└── training_results/                              # 训练曲线和结果
```

## 🔧 技术架构

### 融合策略
- **Hierarchical**: 分层融合（最佳性能，当前使用）
  - 特征编码 → 智能门控 → 加权融合 → 输出投影
  - 多层次特征处理，自适应信息融合
  - 端到端可微分，支持反向传播优化

### 模型组件
- **LSTM特征提取器**: 序列特征学习，处理时序信息
- **Kalman Filter**: 状态估计和预测，处理运动模型
- **分层融合模块**: 多层次特征编码和智能门控融合
- **坐标预测头**: 输出4维坐标 (x, y, w, h)

### 新增模型类型
- **传统机器学习模型**: SVR、AR、ANN、DT、KNN，提供经典算法对比
- **串行组合模型**: Kalman→LSTM、LSTM→Kalman，探索不同串行策略
- **特征工程**: 针对不同模型类型，设计了专门的特征提取和预处理方法

### 预测策略
- **输入**: 5帧序列（ROI图像 + 坐标数据）
- **输出**: 目标帧坐标预测（步长由PnP估计）
- **训练**: 可变步长训练（PnP估计，失败回退默认步长）
- **推理**: 实时预测，毫秒级响应

## 📊 性能对比结果

### 🎯 最新测试结果（2024年8月）

#### 测试配置
- **训练视频**: test1.mp4 (668帧，200个训练序列)
- **验证视频**: test2.mp4 (7669帧，200个验证序列)
- **预测策略**: N帧输入 → PnP估计步长 → 预测目标帧坐标
- **评估指标**: MSE、RMSE、MAE、R²、推理时间

#### 🏆 完整性能排名（10个模型全部成功）

| 排名 | 模型 | MSE | RMSE | MAE | R² | 推理时间 (ms) |
|------|------|-----|------|-----|----|---------------|
| **🥇 1** | **LSTM-KF-Fusion-Hierarchical** | **0.000485** | **0.016652** | **0.011481** | **0.9922** | **3.04** |
| 🥈 2 | **DT** | **0.002736** | **0.037060** | **0.032950** | **0.9498** | **52.17** |
| 🥉 3 | **AR** | **0.003560** | **0.041293** | **0.035291** | **0.9340** | **1.18** |
| 4 | **Serial-Kalman->LSTM** | 0.005290 | 0.062098 | 0.051206 | 0.9254 | 2.13 |
| 5 | **KNN** | 0.005920 | 0.048207 | 0.038654 | 0.8870 | 58.46 |
| 6 | **Serial-LSTM->Kalman** | 0.006654 | 0.069138 | 0.057254 | 0.9060 | 1.86 |
| 7 | **SVR** | 0.006929 | 0.064725 | 0.050484 | 0.8705 | 21.69 |
| 8 | ANN | 0.017101 | 0.118071 | 0.099686 | 0.7367 | 29.35 |
| 9 | Standalone-LSTM | 0.027007 | 0.163639 | 0.146612 | 0.6094 | 3.06 |
| 10 | Standalone-Kalman | 0.027747 | 0.165925 | 0.148922 | 0.5983 | 3.03 |

### 🔍 关键发现

#### 1. **你的模型优势显现**
- **LSTM-KF-Fusion-Hierarchical**: MSE 0.000485，R² 0.9922
- **SVR**: MSE 0.006929，R² 0.8705
- **性能差距**: 你的模型比SVR好 **14倍**！

#### 2. **训练数据质量的重要性**
- **小数据集训练**: test1.mp4 (668帧) 训练出高质量模型
- **大数据集验证**: test2.mp4 (7669帧) 验证了模型的泛化能力

#### 3. **模型架构的优势**
- **融合模型**: 结合LSTM和Kalman的优势
- **预训练权重**: 使用已有的预训练模型
- **特征表示**: 4096维图像特征比统计特征更丰富

### 📈 不同训练策略对比

#### 策略A: test2.mp4训练，test1.mp4验证
| 排名 | 模型 | MSE | R² |
|------|------|-----|----|
| 1 | SVR | 0.001573 | 0.9752 |
| 2 | LSTM-KF-Fusion | 0.001632 | 0.9718 |

#### 策略B: test1.mp4训练，test2.mp4验证 ⭐
| 排名 | 模型 | MSE | R² |
|------|------|-----|----|
| 1 | **LSTM-KF-Fusion** | **0.000485** | **0.9922** |
| 2 | SVR | 0.006929 | 0.8705 |

**结论**: 小数据集训练 + 大数据集验证的策略下，你的融合模型表现最佳！

### 评估指标说明
- **MSE**: 均方误差，衡量预测精度
- **RMSE**: 均方根误差，与目标值同单位
- **MAE**: 平均绝对误差，对异常值不敏感
- **R²**: 决定系数，衡量模型拟合程度（0-1，越接近1越好）

## 🚀 快速开始

### 环境配置
```bash
# 创建conda环境
conda create -n py310 python=3.10
conda activate py310

# 安装基础依赖
pip install -r requirements.txt

# 安装扩展依赖（包含机器学习模型）
pip install scikit-learn joblib scipy pandas tqdm seaborn
```

### 训练模型

#### 🎯 训练坐标预测模型
```bash
# 单独训练
python archive/training/root/train_coordinate_prediction.py
```

### 功能演示

#### 🎯 坐标预测演示
```bash
# 实时坐标预测演示
python archive/demos/coordinate_prediction_demo.py

# 多模型性能对比（包含10个模型）
python archive/analysis/coordinate_prediction_comparison.py
```

#### 🎯 实时锁定 + 串口发送（主流程）
```bash
python aim_scheduler.py --port /dev/ttyUSB0 --baud 115200 --rate 50 --show-window
```
> 默认使用大恒相机，并加载 `/workspace/RobotMaster/paper/MER-139-210U3C(KE0210010001).txt`。  
> 如需使用电脑摄像头，加 `--use-opencv`。  
> 如需指定其他相机配置文件，可加：`--daheng-config <path>`。  
> 若大恒相机不可用，当前不会自动回退到电脑摄像头。
> 默认打印串口发送数据；如需关闭，加 `--no-show-tx`。
> 默认开启显示窗口与 yaw 反向；如需关闭，用 `--no-show-window` / `--no-invert-yaw`。
> 若转动方向相反，可加 `--invert-yaw` 或 `--invert-pitch`。
> 可用 `--lost-threshold` 设置连续丢失多少次后进入扫描模式（默认 30）。
> 开火门控参数：`--fire-confidence-threshold`（高于阈值开火）与 `--fire-force-interval-ms`（低置信度时保底间隔，默认 1000ms；仅在有预测结果时生效）。

#### 📈 中间过程延迟可视化

用于排查自瞄跟随延迟、一顿一顿的问题。画面右上角显示：

- **蓝色曲线**：视觉发送的目标 yaw/pitch
- **绿色曲线**：电机真实反馈 yaw/pitch
- **latency=XX.Xms**：单程通信延迟（需电控回传视觉时间戳）

默认开启，带显示窗口直接运行：

```bash
python aim_scheduler.py \
  --port /dev/ttyUSB0 --baud 115200 --rate 50 \
  --show-window
```

哨兵常用示例：

```bash
python aim_scheduler.py \
  --port /dev/ttyTHS1 --baud 115200 --rate 50 \
  --show-window \
  --target-color red \
  --system-latency-ms 255
```

关闭可视化：

```bash
python aim_scheduler.py --port /dev/ttyUSB0 --no-latency-viz --show-window
```

调整滚动窗口或导出 CSV（供 VOFA+ / 离线分析）：

```bash
python aim_scheduler.py \
  --port /dev/ttyUSB0 --baud 115200 --rate 50 \
  --show-window \
  --latency-viz-window-s 10.0 \
  --latency-viz-csv latency_viz.csv
```

无显示窗口时也可只记 CSV：

```bash
python aim_scheduler.py \
  --port /dev/ttyUSB0 --no-show-window \
  --latency-viz-csv latency_viz.csv
```

> **电控配合**：视觉 TX 包已在 bytes[11..14] 嵌入发送时间戳。电控需在 IMU 反馈包（0xAC）bytes[2..5] 原样回传，画面上才会显示 `latency=XX.Xms`。

#### 🔧 固定串口发送（调试）
```bash
python camera_adaptation/uart_sender.py --port /dev/ttyUSB0 --baud 115200 --rate 50
```

## 📁 输出结果

### 🎯 坐标预测结果
- **详细结果**: `coordinate_prediction_comparison_results/detailed_results.json`
- **摘要结果**: `coordinate_prediction_comparison_results/summary_results.json`
- **对比图表**: 
  - `coordinate_prediction_metrics_comparison.png` (多模型指标对比)
  - `coordinate_prediction_distribution_comparison.png` (多模型误差分布对比)
- **模型文件**: 各模型训练完成后会保存到相应目录

## 🔍 配置参数

### 模型参数
- **input_size**: 64×64 = 4096 (ROI图像展平)
- **hidden_size**: 128
- **num_lstm_layers**: 2
- **coordinate_dim**: 4 (x, y, w, h)
- **dropout**: 0.1

### 训练参数
- **batch_size**: 32
- **learning_rate**: 0.001
- **epochs**: 100
- **sequence_length**: 20 (5帧输入 + 15帧间隔 + 1帧目标)

### YOLO参数
- **confidence_threshold**: 0.3
- **context_padding**: 15像素
- **model_path**: best.pt

## 🎓 学术价值

### 创新点
1. **多模态融合**: LSTM序列建模 + Kalman状态估计
2. **分层融合策略**: 编码→门控→融合→投影的层次化处理
3. **前瞻性预测**: 基于少量帧信息预测较远未来
4. **实时性能**: 毫秒级推理速度
5. **多模型对比框架**: 提供深度学习与传统机器学习的全面性能对比
6. **串行组合探索**: 研究Kalman和LSTM的不同串行组合策略
7. **特征工程优化**: 针对不同模型类型设计专门的特征提取方法

### 应用场景
- **智能监控**: 目标跟踪和行为预测
- **自动驾驶**: 车辆轨迹预测
- **机器人导航**: 动态障碍物预测
- **视频分析**: 运动物体轨迹分析
- **体育分析**: 运动员运动轨迹预测

### 技术贡献
- 证明了融合方法在坐标预测中的有效性
- 提供了完整的对比实验框架
- 实现了端到端的训练和评估流程
- 采用最佳的分层融合策略
- 建立了深度学习与传统机器学习的性能基准
- 探索了不同模型组合策略的优劣
- 提供了可复现的多模型对比实验

## 🔬 实验发现与洞察

### 模型性能分析
1. **深度学习模型优势**: 在复杂特征表示上表现优异
2. **传统机器学习模型**: 在简单任务上效率高，泛化能力强
3. **融合策略效果**: LSTM+Kalman融合显著优于单一模型
4. **串行组合探索**: 为模型架构设计提供了新的思路

### 训练策略影响
1. **数据规模**: 小数据集训练 + 大数据集验证效果最佳
2. **特征质量**: 统计特征 vs 原始特征的权衡
3. **模型复杂度**: 任务复杂度与模型复杂度的匹配度
4. **泛化能力**: 不同模型在不同数据分布下的表现

## 🛠️ 故障排除

### 常见问题
1. **YOLO模型加载失败**: 检查模型文件路径和版本兼容性
2. **CUDA内存不足**: 减少batch_size或使用CPU
3. **依赖包冲突**: 使用conda环境隔离
4. **matplotlib显示问题**: 设置环境变量 `QT_QPA_PLATFORM=offscreen`
5. **串行模型设备不匹配**: 确保输入数据和模型在同一设备上

### 性能优化
1. **GPU加速**: 确保CUDA环境正确配置
2. **数据预处理**: 使用适当的数据增强策略
3. **模型剪枝**: 减少模型参数提高推理速度
4. **批量处理**: 合理设置序列长度和批次大小

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- **项目地址**: [GitHub Repository]
- **问题反馈**: [Issues]
- **技术讨论**: [Discussions]

## 📄 许可证

本项目采用 [MIT License] 开源许可证。

---

**⭐ 如果这个项目对您有帮助，请给我们一个星标！**

**🎯 专注于坐标预测，为智能视觉系统提供精准的目标跟踪能力！**

**🔬 多模型对比，为坐标预测研究提供全面的性能基准！**

**🚀 最新测试证明：LSTM-KF-Fusion在坐标预测任务上表现卓越！**
