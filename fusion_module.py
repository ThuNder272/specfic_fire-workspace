import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """分层融合模块，融合 LSTM 和 Kalman Filter 的输出"""
    
    def __init__(self, hidden_size: int):
        super(FusionModule, self).__init__()
        self.hidden_size = hidden_size
        
        # 分层融合架构
        # LSTM特征编码器
        self.lstm_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Kalman特征编码器
        self.kf_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 融合门控网络
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 输出投影网络
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, h_lstm: torch.Tensor, h_kf: torch.Tensor) -> torch.Tensor:
        """
        前向传播：分层融合 LSTM 和 Kalman Filter 的隐藏状态
        
        Args:
            h_lstm: LSTM 隐藏状态，形状为 (batch_size, hidden_size)
            h_kf: Kalman Filter 隐藏状态，形状为 (batch_size, hidden_size)
            
        Returns:
            h_fused: 融合后的隐藏状态，形状为 (batch_size, hidden_size)
        """
        # 第一阶段：特征编码
        h_lstm_encoded = self.lstm_encoder(h_lstm)
        h_kf_encoded = self.kf_encoder(h_kf)
        
        # 第二阶段：融合门控计算
        combined = torch.cat([h_lstm_encoded, h_kf_encoded], dim=-1)
        gate = self.fusion_gate(combined)
        
        # 第三阶段：加权融合
        h_fused = gate * h_lstm_encoded + (1 - gate) * h_kf_encoded
        
        # 第四阶段：输出投影
        h_final = self.output_projection(h_fused)
        
        return h_final

def test_fusion_module():
    """测试分层融合模块"""
    print("测试分层融合模块...")
    
    batch_size = 4
    hidden_size = 64
    
    # 创建分层融合模块
    fusion_module = FusionModule(hidden_size)
    
    # 创建测试输入
    h_lstm = torch.randn(batch_size, hidden_size)
    h_kf = torch.randn(batch_size, hidden_size)
    
    print(f"LSTM 隐藏状态形状: {h_lstm.shape}")
    print(f"Kalman Filter 隐藏状态形状: {h_kf.shape}")
    
    # 测试融合模块
    print(f"\n测试分层融合模块:")
    h_fused = fusion_module(h_lstm, h_kf)
    print(f"  融合后隐藏状态形状: {h_fused.shape}")
    print(f"  输出范围: [{h_fused.min().item():.4f}, {h_fused.max().item():.4f}]")
    
    print("\n分层融合模块测试完成！")
    
    return fusion_module

if __name__ == "__main__":
    test_fusion_module()
