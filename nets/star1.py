import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        '''add_module是torch.nn.Sequential类的方法之一 用来向顺序容器中添加子模块
        该代码就是将一个名字为conv的子模块添加到了CONVBN中 是一个卷积模块'''
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1) #将权重初始化为1 偏执初始化为0
            torch.nn.init.constant_(self.bn.bias, 0)
# class ConvBN(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         super(ConvBN, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
#         self.with_bn = with_bn
#         if with_bn:
#             self.bn = nn.BatchNorm2d(out_planes)
#             nn.init.constant_(self.bn.weight, 1)  # 将权重初始化为1
#             nn.init.constant_(self.bn.bias, 0)    # 将偏置初始化为0
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.with_bn:
#             x = self.bn(x)
#         return x

# class DownsampleBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, with_bn=True):
#         super().__init__()
#         self.conv1 = ConvBN( in_planes, out_planes, kernel_size, stride, padding, with_bn=with_bn)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear or nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
    # def forward(self, x):
    #     x = self.conv1(x)
    #     return x


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv_7x7 = ConvBN(dim, dim, 5, 1, (5 - 1) // 2, groups=dim, with_bn=True)
        #self.dwconv_3x3 = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=True)
        self.dwconv_5x5 = ConvBN(dim, dim, 3, 1, (3 - 1) // 2, groups=dim, with_bn=True)

        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        #self.f3 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        #self.f4 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 5, 1, (5 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_1 = self.dwconv_7x7(x)
        x_2 = self.dwconv_5x5(x)
        #x_3 = self.dwconv_3x3(x)

        x1,x2= self.f1(x_1),self.f2(x_2)
        x=self.act(x1) * self.act(x2)
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x



class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, **kwargs):
        super().__init__()
        #self.num_classes = num_classes
        self.in_channel = 32
        # stem layer初始化卷积层 从初始的通道数3 变为in-channel
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        #该代码随机生成一个深度丢弃列表 用于后续block的随机深度丢弃
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages 构建多个stage的阶段
        self.stages = nn.ModuleList()
        cur = 0 #用于跟踪在dpr中列表的位置

        #列表的长度确定了一共有几个stage
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer#定义输出通道
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)#降采样
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # # head
        # self.norm = nn.BatchNorm2d(self.in_channel)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(self.in_channel, num_classes)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        feat=[]
        for stage in self.stages:
            x = stage(x)
            feat.append(x)
       #x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return feat[0],feat[1],feat[2],feat[3]

 #我用s1的代码
@register_model
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        url = model_urls['starnet_s2']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        url = model_urls['starnet_s4']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model


# very small networks #
@register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)
if __name__ == '__main__':
    #创建一个startnets1的模型实例
    model=starnet_s1(pretrained=False)
    #print(model)

    input_tensor=torch.randn(1,3,640,640)
    _,out1,out2,out3 = model(input_tensor)
    print(out1.shape)
    print(out2.shape)
    print(out3.shape)
#权重752.pth