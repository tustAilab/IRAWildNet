import numpy as np
import torch
import torch.nn as nn
from nets.star import StarNet,Block
from nets.fusion import DASI
from nets.fusion1 import DASI1
from nets.fusion2 import DASI2
from nets.backbone import Backbone, C2f, Conv,SPPF
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors


'---------------------我的亚像素超分辨重建 可以代替上采样 只要把参数设置为2-------------------------------------------------------'
class espc(nn.Module):
    def __init__(self, upscale_factor, in_channel):
        super(espc, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, in_channel, 5, 1, 2)
        self.conv2 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channel, in_channel * (upscale_factor ** 2), 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.sigmoid(self.pixel_shuffle(x))
        return x


'---------------------------------------------------------------------------------------------------------------------'


class ConvBN1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super(ConvBN1, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_planes)
            nn.init.constant_(self.bn.weight, 1)  # 将权重初始化为1
            nn.init.constant_(self.bn.bias, 0)    # 将偏置初始化为0
        self.weight = self.conv.weight
        self.bias = self.conv.bias if self.conv.bias is not None else None
    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return x

def fuse_conv_and_bn(conv, bn):
    # 混合Conv2d + BatchNorm2d 减少计算量
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # 准备kernel
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # 准备bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class DFL(nn.Module):
    # DFL模块
    # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        # bs, self.reg_max * 4, 8400
        b, c, a = x.shape
        # bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
        # 以softmax的方式，对0~16的数字计算百分比，获得最终数字。
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)




'---------------------------------------------------------------------------------------------------------------------'


# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, input_shape, num_classes, phi, pretrained=False):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
        dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]

        # base_channels       = int(wid_mul * 64)  # 64
        base_channels = 24
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是3, 640, 640
        # -----------------------------------------------#

        # ---------------------------------------------------#
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   48, 80, 80
        #   96, 40, 40
        #   192 20, 20
        # ---------------------------------------------------#
        # self.backbone   = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
        self.backbone = StarNet(24, [2, 2, 8, 3])
        self.sppf= SPPF(base_channels*8, base_channels*8, k=5)
        #----------------DASI特征融合------------------------------------#
        self.dasi1 = DASI1(in_features=base_channels * 8,out_features=base_channels * 8*4)
        self.dasi2 = DASI(in_features=base_channels * 4, out_features=base_channels * 4*4)
        self.dasi3 = DASI(in_features=base_channels * 2, out_features=base_channels * 2*4)
        self.upsample = espc(2, base_channels*2)
        self.dasi1_2 = DASI1(in_features=base_channels * 8,out_features=base_channels * 8*4)
        self.dasi2_2 = DASI(in_features=base_channels * 4, out_features=base_channels * 4*4)
        self.dasi3_2 = DASI2(in_features=base_channels * 2, out_features=base_channels * 2*4)
        #self.dasi4 = DASI(in_feature=base_channels * 1, out_feature=base_channels * 1)


        # ------------------------加强特征提取网络------------------------#
        # # self.upsample   = nn.Upsample(scale_factor=2, mode="nearest") #20 20 192-->40 40 192
        # # self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # self.down_sampler1 = ConvBN1(base_channels*2, base_channels*4, 3, 2, 1) #80 80 48-->40 40 96
        # self.down_sampler2 = ConvBN1(base_channels * 4, base_channels * 8, 3, 2, 1) # 40 40 96-->20 20 192
        # self.bnconv1= ConvBN1(base_channels * 8, base_channels * 4) #concat之后 变成 40 40 192 -->40 40*96
        # self.starblock1=Block(base_channels * 4, mlp_ratio=3, drop_path=0.) # 用一个starblock 40 40 96
        # self.bnconv2= ConvBN1(base_channels *16, base_channels * 8)  #concat之后 20 20 384 --> 20 20 192
        # self.starblock2 = Block(base_channels * 8, mlp_ratio=3, drop_path=0.) #用一个statblock 20 20 192
        # # 1024 * deep_mul + 512, 40, 40 => 512, 40, 40
        # # 192+96 40 40 -->96,40,40



        # ------------------------加强特征提取网络------------------------#

        ch = [base_channels * 2, base_channels * 4, int(base_channels * 8)]
        self.shape = None
        self.nl = len(ch)
        # self.stride     = torch.zeros(self.nl)
        self.stride = torch.tensor(
            [256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))[-3:]])  # forward
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.num_classes = num_classes

        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
        if not pretrained:
            weights_init(self)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        return self

    def forward(self, x):
        #  backbone
        feat0, feat1, feat2, feat3 = self.backbone.forward(x)
        '''------------DASI直接进行特征融合------------------------'''
        P5_SPPF=self.sppf( feat3)
        P5_dasi=self.dasi1(feat3,P5_SPPF,feat2) # 20 20 192
        P4_dasi=self.dasi2(feat2,feat3,feat1) #40 40 96
        P3_dasi=self.dasi3(feat1,feat2,feat0) #80 80 48
        '''------------------------------------------------------'''

        # feat1 48 80 80 feat2 96 40 40 feat3 192 20 20
        # ------------------------加强特征提取网络------------------------#
        P5_SPPF_2=self.sppf(P5_dasi) #20 20 192
        P3_upsamle=self.upsample(P3_dasi) #160 160 48
        P5=self.dasi1_2( P5_dasi,P5_SPPF_2, P4_dasi)

        P4=self.dasi2_2(P4_dasi,P5_dasi,P3_dasi)
        P3=self.dasi3_2(P3_dasi,P4_dasi,P3_upsamle)




        # ------------------------加强特征提取网络------------------------#
        # P3 48, 80, 80
        # P4 96, 40, 40
        # P5 192, 20, 20
        shape = P3.shape  # BCHW

        # P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
        # P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
        # P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
        x = [P3, P4, P5]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400;
        #                                           box self.reg_max * 4, 8400
        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split(
            (self.reg_max * 4, self.num_classes), 1)
        # origin_cls      = [xi.split((self.reg_max * 4, self.num_classes), 1)[1] for xi in x]
        dbox = self.dfl(box)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)