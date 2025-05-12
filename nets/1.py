import torch
import torch.nn as nn
import torch.nn.functional as F

class Bag(nn.Module):
    def __init__(self, channel):
        super(Bag, self).__init__()
        # 使用1x1卷积扩增通道数至channel*3
        self.weight_conv = nn.Conv2d(channel, channel * 3, kernel_size=1)

    def forward(self, p, i, d):
        # 使用1x1卷积生成扩展的权重
        expanded_weights = self.weight_conv(d)

        # 分割权重为三部分，每部分大小为[1, channel, 64, 64]
        weights_p, weights_d, weights_i = torch.chunk(expanded_weights, 3, dim=1)

        # 调整维度以准备在正确的维度上应用softmax
        # 形状转换为[1, 64, 64, channel, 3]
        weights = torch.stack([weights_p, weights_i, weights_d], dim=4).permute(0, 2, 3, 1, 4)

        # 在最后一个维度（不同组的维度）上应用softmax
        softmax_weights = F.softmax(weights, dim=4)

        # 恢复原始维度
        softmax_weights = softmax_weights.permute(0, 3, 1, 2, 4).unbind(dim=4)

        # 解绑后得到归一化的三组权重
        weights_p, weights_i, weights_d = softmax_weights

        # 应用权重并计算加权的输出
        out_p = weights_p * p
        out_i = weights_i * i
        out_d = weights_d * d

        # 返回加权特征图的和
        return out_p + out_i + out_d

# 初始化模型和测试数据
channel = 64
model = Bag(channel)
p = torch.randn(1, channel, 64, 64)
i = torch.randn(1, channel, 64, 64)
d = torch.randn(1, channel, 64, 64)

# 前向传播，获取输出
output = model(p, i, d)
