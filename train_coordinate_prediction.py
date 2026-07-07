#!/usr/bin/env python3
"""
坐标预测模型训练脚本
训练模型预测目标物体的坐标位置 (x, y, w, h)

训练策略：输入前N帧，使用PnP估计的预测帧数作为目标帧。
PnP失败时回退到默认步长，保证训练稳定。
"""

import argparse
import glob
import os
import sys

# 设置Qt环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib字体，确保标题正常显示
plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16
import json
from tqdm import tqdm
import time

from coordinate_prediction_model import CoordinatePredictionModel, CoordinateDataset

class CoordinatePredictionTrainer:
    """坐标预测模型训练器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = {}
        print(f"坐标预测模型训练器初始化完成，设备: {device}")
    
    def create_model(self, input_size=64*64, hidden_size=128, num_lstm_layers=2, dropout=0.1):
        """创建坐标预测模型"""
        print("正在创建坐标预测模型...")
        
        self.model = CoordinatePredictionModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,

            dropout=dropout,
            coordinate_dim=4  # x, y, w, h
        ).to(self.device)
        
        print(f"✓ 成功创建坐标预测模型")
        print(f"   - 输入尺寸: {input_size}")
        print(f"   - 隐藏层尺寸: {hidden_size}")
        print(f"   - LSTM层数: {num_lstm_layers}")
        print(f"   - Dropout率: {dropout}")
        print(f"   - 坐标维度: 4 (x, y, w, h)")
        print(f"   - 融合策略: hierarchical")
        
        return self.model
    
    def create_optimizer(self, learning_rate=0.001, weight_decay=1e-5):
        """创建优化器和学习率调度器"""
        print("正在创建优化器...")
        
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10, 
            #verbose=True
        )
        
        print(f"✓ 成功创建优化器")
        print(f"   - 优化器: Adam")
        print(f"   - 学习率: {learning_rate}")
        print(f"   - 权重衰减: {weight_decay}")
        
        return self.optimizer, self.scheduler
    
    def create_data_loaders(
        self,
        video_path,
        yolo_model_path,
        batch_size=32,
        sequence_length=15,
        num_sequences=1000,
        train_split=0.8,
        input_sequence_length=5,
        default_lead_frames=15,
        min_lead_frames=1,
        max_lead_frames=15,
        bullet_speed_mps=28.0,
        system_latency_s=0.0,
        default_fps=60.0,
        pnp_profile="mer_139_210u3c",
        max_pnp_error=5.0,
        auto_target_type=True,
        target_type="armor_small",
        scale_intrinsics=True,
    ):
        """创建数据加载器"""
        print("正在创建坐标预测数据集...")
        
        try:
            # 创建完整数据集
            full_dataset = CoordinateDataset(
                video_path=video_path,
                yolo_model_path=yolo_model_path,
                sequence_length=sequence_length,
                num_sequences=num_sequences,
                confidence_threshold=0.3,
                context_padding=15,
                input_sequence_length=input_sequence_length,
                default_lead_frames=default_lead_frames,
                min_lead_frames=min_lead_frames,
                max_lead_frames=max_lead_frames,
                bullet_speed_mps=bullet_speed_mps,
                system_latency_s=system_latency_s,
                default_fps=default_fps,
                pnp_profile=pnp_profile,
                max_pnp_error=max_pnp_error,
                auto_target_type=auto_target_type,
                target_type=target_type,
                scale_intrinsics=scale_intrinsics,
            )
            
            # 检查数据集是否为空
            if len(full_dataset) == 0:
                raise ValueError("数据集为空，无法创建数据加载器")
            
            print(f"✓ 数据集创建成功，包含 {len(full_dataset)} 个序列")
            
            # 分割训练集和验证集
            train_size = int(train_split * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            # 确保训练集和验证集都不为空
            if train_size == 0 or val_size == 0:
                print("⚠️ 警告：训练集或验证集为空，调整分割比例...")
                train_size = max(1, len(full_dataset) - 1)
                val_size = len(full_dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=min(batch_size, len(val_dataset)), 
                                  shuffle=False, num_workers=0)
            
            print(f"✓ 数据加载器创建完成")
            print(f"   - 训练集批次数: {len(train_loader)}")
            print(f"   - 验证集批次数: {len(val_loader)}")
            print(f"   - 总序列数: {len(full_dataset)}")
            print(f"   - 训练集大小: {len(train_dataset)}")
            print(f"   - 验证集大小: {len(val_dataset)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            print(f"❌ 创建数据加载器失败: {e}")
            raise
    
    def train_model(self, train_loader, val_loader, num_epochs=100, 
                   early_stopping_patience=20):
        """训练坐标预测模型"""
        print(f"\n🚀 开始训练坐标预测模型")
        print(f"   - 训练轮数: {num_epochs}")
        print(f"   - 早停耐心值: {early_stopping_patience}")
        
        # 训练历史
        train_losses = []
        val_losses = []
        learning_rates = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 损失函数 - 使用MSE损失预测坐标
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for batch_idx, (input_frames, input_coords, target_coords) in enumerate(train_pbar):
                input_frames = input_frames.to(self.device)
                input_coords = input_coords.to(self.device)
                target_coords = target_coords.to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                outputs = self.model(input_frames, input_coords=input_coords)
                predicted_coordinates = outputs['predicted_coordinates']
                
                # 计算坐标预测损失
                loss = criterion(predicted_coordinates, target_coords)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for batch_idx, (input_frames, input_coords, target_coords) in enumerate(val_pbar):
                    input_frames = input_frames.to(self.device)
                    input_coords = input_coords.to(self.device)
                    target_coords = target_coords.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(input_frames, input_coords=input_coords)
                    predicted_coordinates = outputs['predicted_coordinates']
                    
                    # 计算坐标预测损失
                    loss = criterion(predicted_coordinates, target_coords)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            # 学习率调度
            self.scheduler.step(avg_val_loss)
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # 保存最佳模型
                self.save_model(epoch, best_val_loss)
                print(f"✓ 保存最佳模型 (Epoch {epoch+1}, Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
            
            # 打印进度
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"Best Val Loss: {best_val_loss:.6f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Patience: {patience_counter}/{early_stopping_patience}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                print(f"🛑 早停触发，在Epoch {epoch+1}停止训练")
                break
        
        # 保存训练历史
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': learning_rates,
            'best_epoch': epoch - patience_counter + 1,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1
        }
        
        print(f"✅ 坐标预测模型训练完成！")
        print(f"   - 最佳验证损失: {best_val_loss:.6f}")
        print(f"   - 最佳训练轮数: {self.training_history['best_epoch']}")
        print(f"   - 总训练轮数: {self.training_history['total_epochs']}")
        
        return self.training_history
    
    def save_model(self, epoch, val_loss):
        """保存模型"""
        # 创建保存目录
        save_dir = 'coordinate_prediction_models'
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_lstm_layers': self.model.num_lstm_layers,
                'dropout': self.model.dropout,
                'coordinate_dim': self.model.coordinate_dim,
                'fusion_method': 'hierarchical'
            }
        }
        
        # 保存最佳模型
        best_model_path = os.path.join(save_dir, 'Coordinate-Prediction-Model_best.pth')
        torch.save(checkpoint, best_model_path)
        
        # 保存最新检查点
        latest_model_path = os.path.join(save_dir, 'Coordinate-Prediction-Model_latest.pth')
        torch.save(checkpoint, latest_model_path)
        
        print(f"✓ 模型已保存到: {best_model_path}")
    
    def save_training_history(self):
        """保存训练历史"""
        os.makedirs('training_results', exist_ok=True)
        
        # 保存详细历史
        history_path = 'training_results/coordinate_prediction_training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        print(f"✓ 训练历史已保存到: {history_path}")
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        os.makedirs('training_results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Coordinate Prediction Model Training Curves', fontsize=16, fontweight='bold')
        
        # 训练损失
        axes[0, 0].plot(self.training_history['train_losses'], 'b-', linewidth=2, label='Training Loss')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 验证损失
        axes[0, 1].plot(self.training_history['val_losses'], 'r-', linewidth=2, label='Validation Loss')
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[1, 0].plot(self.training_history['learning_rates'], 'g-', linewidth=2, label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Change', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 损失对比
        axes[1, 1].plot(self.training_history['train_losses'], 'b-', linewidth=2, label='Training Loss')
        axes[1, 1].plot(self.training_history['val_losses'], 'r-', linewidth=2, label='Validation Loss')
        axes[1, 1].set_title('Training vs Validation Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = 'training_results/coordinate_prediction_training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 训练曲线图已保存到: {plot_path}")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="使用录制比赛视频训练坐标/LSTM预测模型",
    )
    parser.add_argument("--video-path", default=None, help="训练视频路径")
    parser.add_argument(
        "--use-latest-recording",
        action="store_true",
        default=False,
        help="自动使用 recordings/ 目录下最新的录制视频",
    )
    parser.add_argument("--recordings-dir", default="recordings", help="录制视频目录")
    parser.add_argument(
        "--yolo-model",
        default="best.pt",
        help="检测模型路径，支持普通YOLO权重以及RM四点 .pt/.engine",
    )
    parser.add_argument("--device", default=None, help="训练设备，默认自动选择 cuda/cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--input-seq", type=int, default=5, help="输入序列长度")
    parser.add_argument("--sequence-length", type=int, default=0, help="总序列长度，0表示自动用 input-seq + max-lead-frames")
    parser.add_argument("--min-lead-frames", type=int, default=1, help="最小预测步长")
    parser.add_argument("--max-lead-frames", type=int, default=15, help="最大预测步长")
    parser.add_argument("--default-lead-frames", type=int, default=15, help="PnP失败时默认预测步长")
    parser.add_argument("--num-sequences", type=int, default=2000, help="训练序列数量")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--bullet-speed", type=float, default=27.0, help="训练时使用的弹速(m/s)")
    parser.add_argument("--system-latency-ms", type=float, default=0.0, help="训练时使用的系统时延(ms)")
    parser.add_argument("--default-fps", type=float, default=60.0, help="视频FPS缺失时使用的默认值")
    parser.add_argument("--pnp-profile", default="mer_139_210u3c", help="PnP相机内参配置名")
    parser.add_argument("--max-pnp-error", type=float, default=5.0, help="PnP误差阈值")
    parser.add_argument("--target-type", default="armor_small", help="默认目标类型")
    parser.add_argument("--auto-target-type", action="store_true", default=True, help="自动根据检测类别选择目标类型(1/10为大装甲，默认开启)")
    parser.add_argument("--no-auto-target-type", action="store_false", dest="auto_target_type", help="关闭自动目标类型")
    parser.add_argument("--scale-intrinsics", action="store_true", default=True, help="按视频分辨率缩放内参(默认开启)")
    parser.add_argument("--no-scale-intrinsics", action="store_false", dest="scale_intrinsics", help="关闭内参缩放")
    parser.add_argument("--hidden-size", type=int, default=128, help="隐藏层尺寸")
    parser.add_argument("--num-lstm-layers", type=int, default=2, help="LSTM层数")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    return parser.parse_args()


def _find_latest_recording(recordings_dir: str):
    base_dir = os.path.abspath(recordings_dir)
    patterns = ("*.avi", "*.mp4", "*.mov", "*.mkv")
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(base_dir, pattern)))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _resolve_video_path(args) -> str:
    if args.video_path:
        video_path = os.path.abspath(args.video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"训练视频不存在: {video_path}")
        return video_path

    if args.use_latest_recording:
        latest = _find_latest_recording(args.recordings_dir)
        if latest is None:
            raise FileNotFoundError(
                f"在录制目录中未找到视频: {os.path.abspath(args.recordings_dir)}"
            )
        print(f"🎞️ 使用最新录制视频: {latest}")
        return latest

    fallback = os.path.abspath("test2.avi")
    if os.path.exists(fallback):
        print(f"⚠️ 未指定训练视频，回退到默认样例: {fallback}")
        return fallback

    latest = _find_latest_recording(args.recordings_dir)
    if latest is not None:
        print(f"⚠️ 未指定训练视频，自动使用最新录制视频: {latest}")
        return latest

    raise FileNotFoundError(
        "未找到训练视频，请通过 --video-path 指定，或先用 --record-video 录制比赛视频"
    )


def main():
    args = _parse_args()

    print("🚀 坐标预测模型训练")
    print("=" * 60)

    video_path = _resolve_video_path(args)
    yolo_model_path = os.path.abspath(args.yolo_model)
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"YOLO模型不存在: {yolo_model_path}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_sequence_length = max(1, int(args.input_seq))
    min_lead_frames = max(0, int(args.min_lead_frames))
    max_lead_frames = max(min_lead_frames, int(args.max_lead_frames))
    default_lead_frames = int(args.default_lead_frames)
    if args.sequence_length and int(args.sequence_length) > 0:
        sequence_length = max(int(args.sequence_length), input_sequence_length + max_lead_frames)
    else:
        sequence_length = input_sequence_length + max_lead_frames

    batch_size = max(1, int(args.batch_size))
    num_sequences = max(1, int(args.num_sequences))
    num_epochs = max(1, int(args.epochs))
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    bullet_speed_mps = max(1e-3, float(args.bullet_speed))
    system_latency_s = max(0.0, float(args.system_latency_ms) / 1000.0)
    default_fps = max(1.0, float(args.default_fps))
    pnp_profile = args.pnp_profile
    max_pnp_error = float(args.max_pnp_error)
    auto_target_type = bool(args.auto_target_type)
    scale_intrinsics = bool(args.scale_intrinsics)

    input_size = 64 * 64
    hidden_size = max(1, int(args.hidden_size))
    num_lstm_layers = max(1, int(args.num_lstm_layers))
    dropout = max(0.0, float(args.dropout))

    print(f"📹 视频文件: {video_path}")
    print(f"🤖 YOLO模型: {yolo_model_path}")
    print(f"🖥️  训练设备: {device}")
    print("⚙️  训练参数:")
    print(f"   - 批次大小: {batch_size}")
    print(f"   - 输入帧数: {input_sequence_length}")
    print(f"   - 序列长度: {sequence_length}")
    print(
        f"   - 预测步长: {min_lead_frames}~{max_lead_frames} (默认 {default_lead_frames})"
    )
    print(f"   - 训练序列数: {num_sequences}")
    print(f"   - 训练轮数: {num_epochs}")
    print(f"   - 学习率: {learning_rate}")
    print(f"   - 权重衰减: {weight_decay}")
    print(f"   - 弹速: {bullet_speed_mps} m/s")
    print(f"   - 系统延迟: {system_latency_s:.3f} s")
    print(f"   - FPS默认值: {default_fps}")
    print(f"   - PnP相机内参: {pnp_profile}")
    print(f"   - PnP误差阈值: {max_pnp_error}")
    print(f"   - 自动目标类型: {auto_target_type}")
    print(f"   - 内参缩放: {scale_intrinsics}")
    print("🤖 模型参数:")
    print(f"   - 输入尺寸: {input_size}")
    print(f"   - 隐藏层尺寸: {hidden_size}")
    print(f"   - LSTM层数: {num_lstm_layers}")
    print(f"   - Dropout率: {dropout}")
    print("🎯 预测目标: 目标物体坐标 (x, y, w, h)")
    print(f"📊 训练策略: PnP估计预测步长，基于前{input_sequence_length}帧预测目标帧")

    try:
        trainer = CoordinatePredictionTrainer(device=device)

        trainer.create_model(
            input_size=input_size,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
        )

        trainer.create_optimizer(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        train_loader, val_loader = trainer.create_data_loaders(
            video_path=video_path,
            yolo_model_path=yolo_model_path,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_sequences=num_sequences,
            input_sequence_length=input_sequence_length,
            default_lead_frames=default_lead_frames,
            min_lead_frames=min_lead_frames,
            max_lead_frames=max_lead_frames,
            bullet_speed_mps=bullet_speed_mps,
            system_latency_s=system_latency_s,
            default_fps=default_fps,
            pnp_profile=pnp_profile,
            max_pnp_error=max_pnp_error,
            auto_target_type=auto_target_type,
            target_type=args.target_type,
            scale_intrinsics=scale_intrinsics,
        )

        training_history = trainer.train_model(
            train_loader,
            val_loader,
            num_epochs=num_epochs,
        )

        trainer.save_training_history()

        print(f"\n🎉 坐标预测模型训练完成！")
        print(f"📁 模型保存在: coordinate_prediction_models/")
        print(f"📊 训练结果保存在: training_results/")
        print(f"\n📈 训练结果摘要:")
        print(f"   - 最佳验证损失: {training_history['best_val_loss']:.6f}")
        print(f"   - 最佳训练轮数: {training_history['best_epoch']}")
        print(f"   - 总训练轮数: {training_history['total_epochs']}")
        print(f"\n💡 接下来您可以:")
        print(f"   1. 使用训练好的模型进行坐标预测")
        print(f"   2. 评估坐标预测精度")
        print(f"   3. 可视化预测轨迹")

    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
