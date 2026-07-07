import torch
import torch.nn as nn
from typing import Tuple

class AdaptiveKalmanFilter(nn.Module):
    """自适应卡尔曼滤波器"""
    
    def __init__(self, hidden_size: int):
        super(AdaptiveKalmanFilter, self).__init__()
        self.hidden_size = hidden_size
        
        # 可学习的噪声参数
        self.log_process_noise = nn.Parameter(torch.tensor(-2.0))
        self.log_measurement_noise = nn.Parameter(torch.tensor(-2.0))
        
        # 可学习的状态转移矩阵
        self.state_transition = nn.Parameter(torch.eye(hidden_size) * 0.95)
        
        # 初始化误差协方差矩阵
        self.P_init = torch.eye(hidden_size) * 1.0
    
    def forward(self, h_prev: torch.Tensor, h_obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h_prev: 前一时刻隐藏状态
            h_obs: 观测值
            
        Returns:
            h_kf: 卡尔曼滤波后的隐藏状态
        """
        batch_size = h_prev.size(0)
        
        # 计算噪声协方差矩阵
        Q = torch.eye(self.hidden_size, device=h_prev.device) * torch.exp(self.log_process_noise)
        R = torch.eye(self.hidden_size, device=h_prev.device) * torch.exp(self.log_measurement_noise)
        
        # 使用初始化的 P 矩阵
        P = self.P_init.to(h_prev.device)
        
        # 状态预测
        h_pred = torch.mm(h_prev, self.state_transition.t())
        P_pred = torch.mm(torch.mm(self.state_transition, P), self.state_transition.t()) + Q
        
        # 扩展到批次维度
        P_pred = P_pred.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 计算卡尔曼增益
        P_plus_R = P_pred + R.unsqueeze(0).expand(batch_size, -1, -1)
        
        try:
            L = torch.linalg.cholesky(P_plus_R)
            P_plus_R_inv = torch.cholesky_inverse(L)
        except:
            P_plus_R_inv = torch.pinverse(P_plus_R)
        
        K = torch.bmm(P_pred, P_plus_R_inv)
        
        # 状态修正
        innovation = h_obs - h_pred
        h_kf = h_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
        
        return h_kf

def test_kalman_filter():
    """测试卡尔曼滤波模块"""
    print("测试卡尔曼滤波模块...")
    
    batch_size = 4
    hidden_size = 64
    
    kf = AdaptiveKalmanFilter(hidden_size)
    h_prev = torch.randn(batch_size, hidden_size)
    h_obs = torch.randn(batch_size, hidden_size)
    
    print(f"前一时刻隐藏状态形状: {h_prev.shape}")
    print(f"观测值形状: {h_obs.shape}")
    
    h_kf = kf(h_prev, h_obs)
    
    print(f"卡尔曼滤波后隐藏状态形状: {h_kf.shape}")
    print(f"过程噪声参数: {torch.exp(kf.log_process_noise):.4f}")
    print(f"测量噪声参数: {torch.exp(kf.log_measurement_noise):.4f}")
    
    print("卡尔曼滤波模块测试完成！")
    
    return kf

if __name__ == "__main__":
    test_kalman_filter()
