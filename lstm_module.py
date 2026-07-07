import torch
import torch.nn as nn
from typing import Tuple

class LSTMFeatureExtractor(nn.Module):
    """LSTM 特征提取器"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.1):
        super(LSTMFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_size, input_size)
        self.activation = nn.Sigmoid()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入序列，形状为 (batch_size, sequence_length, input_size)
            
        Returns:
            h_lstm: LSTM 隐藏状态
            predicted_frame: 预测的下一帧
        """
        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 获取最后时刻的隐藏状态
        h_lstm = h_n[-1]  # 最后一层的最后时刻
        
        # 预测下一帧
        predicted_frame = self.output_projection(h_lstm)
        predicted_frame = self.activation(predicted_frame)
        
        return h_lstm, predicted_frame

def test_lstm_module():
    """测试 LSTM 模块"""
    print("测试 LSTM 模块...")
    
    batch_size = 4
    sequence_length = 8
    input_size = 64 * 64
    hidden_size = 128
    
    lstm_extractor = LSTMFeatureExtractor(input_size, hidden_size, num_layers=2)
    x = torch.randn(batch_size, sequence_length, input_size)
    
    print(f"输入形状: {x.shape}")
    
    h_lstm, predicted_frame = lstm_extractor(x)
    
    print(f"LSTM 隐藏状态形状: {h_lstm.shape}")
    print(f"预测帧形状: {predicted_frame.shape}")
    print("LSTM 模块测试完成！")
    
    return lstm_extractor

if __name__ == "__main__":
    test_lstm_module()
