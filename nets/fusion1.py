import torch
import torch.nn as nn
import torch.nn.functional as F

class Bag(nn.Module):
    def __init__(self, channel):
        super(Bag, self).__init__()
        # 使用1x1卷积扩增通道数至channel*3
        self.weight_conv = nn.Conv2d(channel, channel * 3, kernel_size=1)

    def forward(self, p, i, d):
        # 使用1x1卷积扩展三个组的通道生成扩展的权重
        expanded_weights = self.weight_conv(d)

        # 分割权重为三部分，每部分大小为[1, channel, h,w]
        weights_p, weights_d, weights_i = torch.chunk(expanded_weights, 3, dim=1)

        # 调整维度以准备在正确的维度上应用softmax
        # 形状转换为[1, h,w, channel, 3]
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

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class DASI1(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag(in_features)
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.conv = nn.Sequential(
             conv_block(in_features = out_features // 2,
                        out_features = out_features // 4,
                        kernel_size=(1, 1),
                        padding=(0, 0),
                        norm_type=None,
                        activation=False)
         )
         self.bns = nn.BatchNorm2d(out_features)

         self.skips = conv_block(in_features=in_features,
                                                out_features=out_features,
                                                kernel_size=(1, 1),
                                                padding=(0, 0),
                                                norm_type=None,
                                                activation=False)
         self.skips_2 = conv_block(in_features=in_features,
                                 out_features=out_features,
                                 kernel_size=(1, 1),
                                 padding=(0, 0),
                                 norm_type=None,
                                 activation=False)#1
         self.skips_3 = nn.Conv2d(in_features//2, out_features,
                                  kernel_size=3, stride=2, dilation=2, padding=2)
         # self.skips_3 = nn.Conv2d(in_features//2, out_features,
         #                          kernel_size=3, stride=2, dilation=1, padding=1)
         self.relu = nn.ReLU()
         self.fc = nn.Conv2d(out_features, in_features, kernel_size=1, bias=False)

         self.gelu = nn.GELU()
    def forward(self, x , x_low, x_high):
        if x_high != None:
            x_high = self.skips_3(x_high)
            x_high = torch.chunk(x_high, 4, dim=1)
        if x_low != None:
            x_low = self.skips_2(x_low)
            x_low = F.interpolate(x_low, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=True)
            x_low = torch.chunk(x_low, 4, dim=1)
        x_skip = self.skips(x)
        x = self.skips(x)
        x = torch.chunk(x, 4, dim=1)
        if x_high == None:
            x0 = self.conv(torch.cat((x[0], x_low[0]), dim=1))
            x1 = self.conv(torch.cat((x[1], x_low[1]), dim=1))
            x2 = self.conv(torch.cat((x[2], x_low[2]), dim=1))
            x3 = self.conv(torch.cat((x[3], x_low[3]), dim=1))
        elif x_low == None:
            x0 = self.conv(torch.cat((x[0], x_high[0]), dim=1))
            x1 = self.conv(torch.cat((x[0], x_high[1]), dim=1))
            x2 = self.conv(torch.cat((x[0], x_high[2]), dim=1))
            x3 = self.conv(torch.cat((x[0], x_high[3]), dim=1))
        else:
            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])
        #

        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        x += x_skip
        x = self.bns(x)
        x = self.fc(x)
        x = self.relu(x)

        return x
if __name__ == '__main__':
    # 定义输入数据
    batch_size = 1
    channels = 3
    height = 64
    width = 64
    x = torch.randn(batch_size, channels, height, width)
    x_low = torch.randn(batch_size, channels , height , width )
    x_high = torch.randn(batch_size, channels // 2, height * 2, width * 2)

    # 实例化 DASI 模块
    dasinet = DASI1(channels, channels * 4)

    # 打印输入和输出的形状
    output = dasinet(x, x_low, x_high)
    print("输入 x 的形状:", x.shape)
    print("输入 x_low 的形状:", x_low.shape)
    print("输入 x_high 的形状:", x_high.shape)
    print("输出的形状:", output.shape)