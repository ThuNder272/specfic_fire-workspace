import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from ultralytics import YOLO

class VideoFrameExtractor:
    """视频帧提取器（集成目标检测）"""
    
    def __init__(self, video_path: str, target_size: Tuple[int, int] = (64, 64), 
                 yolo_model_path: Optional[str] = None, confidence_threshold: float = 0.5,
                 context_padding: int = 10):
        self.video_path = video_path
        self.target_size = target_size
        self.confidence_threshold = confidence_threshold
        self.context_padding = context_padding  # 上下文填充像素数
        self.frames = []
        self.detection_masks = []  # 存储检测掩码
        
        # 加载YOLO模型
        if yolo_model_path and os.path.exists(yolo_model_path):
            self.yolo_model = self._load_yolo_model(yolo_model_path)
            if self.yolo_model is not None:
                self.use_detection = True
            else:
                self.use_detection = False
        else:
            self.yolo_model = None
            self.use_detection = False
            print("未使用目标检测，将处理整个帧")
        
        self.extract_frames()
    
    def _load_yolo_model(self, yolo_model_path):
        """仅使用 ultralytics v10 加载模型（如果非 v10 会打印警告）。
        保留 mock 模型作为测试兜底。
        """
        print(f"正在加载 YOLO v10 模型: {yolo_model_path}")
        try:
            # 只使用 ultralytics 的 YOLO 接口（v10）
            from ultralytics import YOLO as UltraYOLO
            yolo_model = UltraYOLO(yolo_model_path)
            # 记录为 v10（如果 ultralytics 版本不是 10.x，会提示但仍尝试使用）
            try:
                import ultralytics as _ultra
                ver = getattr(_ultra, '__version__', None)
                if ver is not None and str(ver).split('.')[0] != '10':
                    print(f"警告: 检测到 ultralytics 版本 {ver}，但当前适配为 v10；可能存在兼容性问题。")
            except Exception:
                pass
            self.yolo_version = 'v10'
            print("✓ 成功使用 ultralytics 加载模型 (v10)")
            return yolo_model
        except Exception as e:
            print(f"ultralytics v10 加载失败: {e}")

        # 如果加载失败，尝试创建一个简单的模拟模型用于测试
        print("⚠️ 无法加载 ultralytics v10 模型，创建模拟模型用于测试...")
        try:
            yolo_model = self._create_mock_yolo_model()
            self.yolo_version = 'mock'
            print("✓ 成功创建模拟YOLO模型")
            return yolo_model
        except Exception as e:
            print(f"模拟模型创建失败: {e}")

        print(f"❌ 无法加载 YOLO v10 模型。请检查模型文件: {yolo_model_path}")
        return None
    
    def _create_mock_yolo_model(self):
        """创建一个模拟的YOLO模型用于测试"""
        class MockYOLOModel:
            def __init__(self):
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            def __call__(self, img, *args, **kwargs):
                # 返回模拟的检测结果
                # 假设图像中心有一个目标
                h, w = img.shape[:2]
                
                # 创建模拟的边界框 (x1, y1, x2, y2, conf, cls)
                x1, y1 = w * 0.3, h * 0.3
                x2, y2 = w * 0.7, h * 0.7
                conf = 0.8
                cls = 0
                
                # 创建模拟结果对象
                class MockResult:
                    def __init__(self):
                        self.pred = [torch.tensor([[x1, y1, x2, y2, conf, cls]], device=self.device)]
                
                result = MockResult()
                return result
        
        return MockYOLOModel()
    
    def extract_frames(self):
        """提取视频中的所有帧（集成目标检测）"""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 如果使用目标检测，创建检测掩码
            if self.use_detection:
                detection_mask = self._create_detection_mask(frame)
                self.detection_masks.append(detection_mask)
            
            # 转换为灰度图并调整大小
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, self.target_size)
            
            # 如果使用目标检测，应用掩码和上下文
            if self.use_detection:
                resized_mask = cv2.resize(self.detection_masks[-1], self.target_size)
                # 应用掩码，但保留一些上下文信息
                resized_frame = resized_frame * (resized_mask / 255.0)
                # 添加轻微的上下文信息（非检测区域保留10%的原始信息）
                context_mask = 1.0 - (resized_mask / 255.0) * 0.9
                resized_frame = resized_frame + (resized_frame * context_mask * 0.1)
            
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            self.frames.append(normalized_frame)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"已提取 {frame_count} 帧")
        
        cap.release()
        print(f"总共提取了 {len(self.frames)} 帧")
        if self.use_detection:
            print(f"已应用目标检测掩码，只关注检测到的物体区域")
    
    def _create_detection_mask(self, frame: np.ndarray) -> np.ndarray:
        """创建目标检测掩码（兼容多版本 YOLO）"""
        try:
            # 创建掩码
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # 统一调用解析器，返回 [(x1,y1,x2,y2,conf,cls), ...]
            detections = self._run_yolo_model(frame)
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # 确保坐标在图像范围内
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    # 在掩码上填充检测区域（包含上下文）
                    y1_pad = max(0, y1 - self.context_padding)
                    y2_pad = min(frame.shape[0], y2 + self.context_padding)
                    x1_pad = max(0, x1 - self.context_padding)
                    x2_pad = min(frame.shape[1], x2 + self.context_padding)
                    mask[y1_pad:y2_pad, x1_pad:x2_pad] = 255

            return mask
        except Exception as e:
            print(f"目标检测失败: {e}")
            # 如果检测失败，返回全1掩码（处理整个帧）
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255

    def _run_yolo_model(self, frame: np.ndarray):
        """仅解析 ultralytics v10 的输出格式，返回列表：[(x1,y1,x2,y2,conf,cls), ...]。
        对于 mock 模型（返回 results.pred）也保持兼容。
        """
        preds = []
        try:
            # ultralytics v10: model(frame) -> Results or list[Results]
            try:
                results = self.yolo_model(frame)
            except Exception as e:
                # 若模型不可调用则返回空
                print(f"调用 YOLO 模型失败: {e}")
                return preds

            res_list = results if isinstance(results, (list, tuple)) else [results]
            for res in res_list:
                # v10 风格：res.boxes 存在且包含 xyxy, conf, cls
                if hasattr(res, 'boxes') and res.boxes is not None:
                    try:
                        # xyxy 可能是 Tensor
                        if hasattr(res.boxes, 'xyxy'):
                            xyxy = res.boxes.xyxy.cpu().numpy()
                            # conf / cls
                            confs = None
                            clss = None
                            if hasattr(res.boxes, 'conf'):
                                confs = res.boxes.conf.cpu().numpy()
                            if hasattr(res.boxes, 'cls'):
                                clss = res.boxes.cls.cpu().numpy()

                            for i, box in enumerate(xyxy):
                                x1, y1, x2, y2 = box[:4]
                                conf = float(confs[i]) if confs is not None else 1.0
                                cls = int(clss[i]) if clss is not None else 0
                                preds.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
                            continue
                    except Exception:
                        # 跳过当前 result
                        continue

                # 兼容 mock：如果结果里有 pred（yolov5 风格的 mock），解析 pred
                if hasattr(res, 'pred') and res.pred is not None:
                    try:
                        for det in res.pred:
                            if det is None:
                                continue
                            det_np = det.cpu().numpy()
                            for d in det_np:
                                if d.shape[0] >= 6:
                                    x1, y1, x2, y2, conf, cls = d[:6]
                                else:
                                    x1, y1, x2, y2 = d[:4]
                                    conf = float(d[4]) if d.shape[0] > 4 else 1.0
                                    cls = int(d[5]) if d.shape[0] > 5 else 0
                                preds.append((int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)))
                        continue
                    except Exception:
                        continue

        except Exception as e:
            print(f"_run_yolo_model 错误: {e}")

        return preds
    
    def get_frame_sequence(self, sequence_length: int = 10) -> List[np.ndarray]:
        """获取帧序列"""
        if len(self.frames) < sequence_length:
            raise ValueError(f"视频帧数 ({len(self.frames)}) 少于序列长度 ({sequence_length})")
        
        start_idx = np.random.randint(0, len(self.frames) - sequence_length + 1)
        return self.frames[start_idx:start_idx + sequence_length]

class VideoDataset(Dataset):
    """视频数据集（支持目标检测）"""
    
    def __init__(self, video_path: str, sequence_length: int = 10, num_sequences: int = 1000,
                 yolo_model_path: Optional[str] = None, confidence_threshold: float = 0.5,
                 context_padding: int = 10):
        self.extractor = VideoFrameExtractor(video_path, yolo_model_path=yolo_model_path, 
                                           confidence_threshold=confidence_threshold,
                                           context_padding=context_padding)
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        
        # 生成序列
        self.sequences = []
        for _ in range(num_sequences):
            try:
                sequence = self.extractor.get_frame_sequence(sequence_length)
                self.sequences.append(sequence)
            except ValueError:
                continue
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence_tensor = torch.FloatTensor(sequence)
        
        # 输入序列和目标帧
        input_sequence = sequence_tensor[:-1]  # 前 sequence_length-1 帧
        target_frame = sequence_tensor[-1]     # 最后一帧
        
        # 展平帧数据
        input_sequence = input_sequence.view(input_sequence.size(0), -1)
        target_frame = target_frame.view(-1)
        
        return input_sequence, target_frame

def create_data_loader(video_path: str, batch_size: int = 32, sequence_length: int = 10, 
                      num_sequences: int = 1000, train_split: float = 0.8,
                      yolo_model_path: Optional[str] = None, confidence_threshold: float = 0.5,
                      context_padding: int = 10) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器（支持目标检测和上下文）"""
    full_dataset = VideoDataset(video_path, sequence_length, num_sequences, 
                               yolo_model_path=yolo_model_path, confidence_threshold=confidence_threshold,
                               context_padding=context_padding)
    
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def visualize_sequence(sequence: torch.Tensor, save_path: str = None):
    """可视化帧序列"""
    sequence_np = sequence.numpy()
    num_frames = sequence_np.shape[0]
    
    fig, axes = plt.subplots(1, num_frames, figsize=(3*num_frames, 3))
    if num_frames == 1:
        axes = [axes]
    
    for i in range(num_frames):
        axes[i].imshow(sequence_np[i], cmap='gray')
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"序列图像已保存到: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    video_path = "your_video.mp4"
    
    if os.path.exists(video_path):
        print("正在处理视频...")
        train_loader, val_loader = create_data_loader(video_path, batch_size=16, sequence_length=10, num_sequences=500)
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 可视化第一个序列
        for batch_idx, (input_sequences, target_frames) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"输入序列形状: {input_sequences.shape}")
                print(f"目标帧形状: {target_frames.shape}")
                
                first_sequence = input_sequences[0].view(-1, 64, 64)
                visualize_sequence(first_sequence, "sample_sequence.png")
                break
    else:
        print(f"视频文件不存在: {video_path}")
