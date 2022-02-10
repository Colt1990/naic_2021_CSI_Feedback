#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
Note:
    1.This file is used for designing the structure of encoder and decoder.
    2.The neural network structure in this model file is CsiNet, more details about CsiNet can be found in [1].
[1] C. Wen, W. Shih and S. Jin, "Deep Learning for Massive MIMO CSI Feedback", in IEEE Wireless Communications Letters, vol. 7, no. 5, pp. 748-751, Oct. 2018, doi: 10.1109/LWC.2018.2818160.
    3.The output of the encoder must be the bitstream.
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as FF
from torch.utils.data import Dataset
from collections import OrderedDict
import math

#This part implement the quantization and dequantization operations.
#The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32) #torch.float16


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its B bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2) / ctx.constant
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for B time.
        b, c = grad_output.shape
        grad_output = grad_output.unsqueeze(2) / ctx.constant
        grad_bit = grad_output.expand(b, c, ctx.constant)
        return torch.reshape(grad_bit, (-1, c * ctx.constant)), None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out

    
def VQQuant(x,codebook):
    B = len(x)
    x = x.unsqueeze(-1)
    _,_,_,codenum = codebook.size()
    num_quan_bits = 4
    if codenum == 4:
        num_quan_bits = 2
    if codenum == 8:
        num_quan_bits = 3

    distance = (x - codebook)**2
    distance = distance.detach()
    distance = torch.sum(distance,dim = 2,keepdims = True)

    min_dis = torch.min(distance,dim = -1,keepdims = True)[1]

    out = min_dis.view(B,-1)
    out = Num2Bit(out,num_quan_bits)
    return out


def VQDeQuant(x,codebook):
    # The size of x is B,F;
    _,_,_,codenum = codebook.size()
    num_quan_bits = 4
    if codenum == 4:
        num_quan_bits = 2
    if codenum == 8:
        num_quan_bits = 3

    x = Bit2Num(x, num_quan_bits)
    codebook = codebook.squeeze(0) # Shape is 1,Feed,C,Seed
    F,C,S = codebook.size()
    codebook = codebook.permute(0,2,1).reshape(-1,C)

    B,F = x.size()
    x = x + torch.arange(F).unsqueeze(0).to(x.device) * S
    x = x.long()
    x = x.view(-1)
    x = torch.index_select(codebook, 0, x).view(B,-1)

    return x    
    
    

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


def conv3x3_bn(in_channels, out_channel, stride=1):
    return nn.Sequential(OrderedDict([
        ("conv3x3", nn.Conv2d(in_channels, out_channel, kernel_size=3,
                              stride=stride, padding=1, groups=1, bias=False)),
        ("bn", nn.BatchNorm2d(num_features=out_channel))
    ]))


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)), #
            ('bn', nn.BatchNorm2d(out_planes))
        ]))


class CRBlock(nn.Module):
    def __init__(self):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(16, 128, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('gelu1',nn.GELU()),
            ('conv1x9', ConvBN(128, 128, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('gelu2',nn.GELU()),
           ('conv7x7',ConvBN(128, 128, 7, groups=4 * 16)),
           ('relu3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('gelu3',nn.GELU()),
            ('conv9x1', ConvBN(128, 128, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(16, 128, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('gelu1',nn.GELU()),
#             ('conv7x7',ConvBN(128, 128, 7, groups=4 * 16)),
#            ('relu3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(128, 128, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(128 * 2, 16, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
#         self.relu = nn.GELU()

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)

        out = self.relu(out + identity)
        return out







class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class RefineBlock(nn.Module):
    def __init__(self):
        super(RefineBlock, self).__init__()
        self.conv1_bn = conv3x3_bn(2, 8)
        self.conv2_bn = conv3x3_bn(8, 16)
        self.conv3_bn = conv3x3_bn(16, 2)
        self.activation = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.activation(self.conv1_bn(x))
        residual = self.activation(self.conv2_bn(residual))
        residual = self.conv3_bn(residual)

        return self.activation(residual + identity)


class TinyRefineBlock(nn.Module):
    r"""
    This is headC for BCsiNet. Residual architecture is included.
    """
    def __init__(self):
        super(TinyRefineBlock, self).__init__()
        self.conv1_bn = conv3x3_bn(2, 4)
        self.conv2_bn = conv3x3_bn(4, 2)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.relu(self.conv1_bn(x))
        residual = self.conv2_bn(residual)

        return self.relu(residual + identity)

# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits):
#         super(Encoder, self).__init__()
        
#         total_size, in_channel, w, h = 32256, 2, 126, 128
#         self.encoder1 = nn.Sequential(OrderedDict([
#             ("conv3x3_bn", ConvBN(in_channel, 256, 3)),
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv1x9_bn", ConvBN(256, 256, [1, 9])),
#             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv9x1_bn", ConvBN(256, 256, [9, 1])),
#         ]))
#         self.encoder2 = ConvBN(in_channel, 128,1)
# #         self.encoder2 = nn.Sequential(OrderedDict([
# #             ("conv3x3_bn1", ConvBN(in_channel, 32, 3)),
# #             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv3x3_bn2", ConvBN(32, 64, 3)),
# #             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv3x3_bn3", ConvBN(64, 32, 3)),
# #             ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv3x3_bn4", ConvBN(32, 16, 3)),
# #             ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #         ]))
        
#         self.encoder_conv = nn.Sequential(OrderedDict([
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv1x1_bn", ConvBN(512, 2, 1)),
#            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#         ]))
#         self.conv3 = ConvBN(128,2,3)
#         self.sa = SpatialGate()
#         self.se = SELayer(16)
#         self.encoder_fc = nn.Linear(total_size, int(feedback_bits // self.B))
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)

#     def forward(self, x):
#         n, c, h, w = x.detach().size()
# #         encode1 = self.encoder1(x)
# #         encode1 = self.sa(encode1)
#         encode2 = self.encoder2(x)
#         #import pdb;pdb.set_trace()
#         encode2 = self.sa(encode2)
#         #out = torch.cat((encode1, encode2), dim=1)
#         #out = self.encoder_conv(out)
#         out = encode2
#         out = self.conv3(out)
        
#         #
#         out = out.view(n, -1)
#         #out = out.unsqueeze(2) #[1,2048,1]
#         out = self.encoder_fc(out) # [1,2048/cr,1]
#         out = self.sig(out)
#         out = self.quantize(out)

#         return out


# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits,num_refinenet=6):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)

#         self.sig = nn.Sigmoid()

#         total_size, in_channel, w, h = 32256, 2, 126, 128
#         self.replace_dfc = nn.ConvTranspose1d(int(feedback_bits // self.B),total_size,1)
#         decoder = OrderedDict([
#             ("conv5x5_bn", ConvBN(2, 2, 5)),
#             ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("CRBlock1", CRBlock()),
#             ("CRBlock2", CRBlock()),
# #             ("conv3x3", conv3x3(2, 2)),
#         ])
        
#         self.decoder_feature = nn.Sequential(decoder)
#         self.sigmoid = nn.Sigmoid()
#         self.hsig= hsigmoid()
#         self.decoder_fc = nn.Linear(int(feedback_bits // self.B), total_size)

#         for m in self.modules():
#             if isinstance(m, (nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         c,h,w = 2,126,128
        
#         out = self.dequantize(x) 
#         out = out.view(-1, int(self.feedback_bits / self.B))
#         #print(out.shape)
#         result = self.decoder_fc(out)
#         result = result.view(-1,2, 126, 128)
        
#         allout = []
#         for i in range(8):
            
#             result1 = self.decoder(result[i*8:(i+1)*8])
#             out = self.final_layer(result1)
#             out = self.hsig(out)
#             allout.append(out)
#         out = torch.cat(allout,0)
#         return out


# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         self.encoder1 = nn.Sequential(OrderedDict([
#             ("conv3x3_bn", ConvBN(2, 128, 3)),
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv1x9_bn", ConvBN(128, 128, [1, 9])),
#             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv9x1_bn", ConvBN(128, 128, [9, 1])),
#         ]))
#         self.encoder2 = ConvBN(2, 128, 3)
#         self.encoder_conv = nn.Sequential(OrderedDict([
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv1x1_bn", ConvBN(128*2, 2, 1)),
#             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#         ]))

#         self.fc = nn.Linear(32256, int(feedback_bits / self.B))
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 

#     def forward(self, x):
#         encode1 = self.encoder1(x)
#         encode2 = self.encoder2(x)
#         out = torch.cat((encode1, encode2), dim=1)
#         out = self.encoder_conv(out)
#         out = out.view(-1, 32256)
#         out = self.fc(out)
#         out = self.sig(out)
#         if self.quantization:
#             out = self.quantize(out)
#         else:
#             out = out
#         return out


# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         self.fc = nn.Linear(int(feedback_bits / self.B), 32256)
#         decoder = OrderedDict([
#             ("conv5x5_bn", ConvBN(2, 128, 5)),
#             ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("CRBlock1", CRBlock()),
#             ("CRBlock2", CRBlock()),
#         ])
#         self.decoder_feature = nn.Sequential(decoder)
#         self.out_cov = conv3x3(128, 2)
#         self.sig = nn.Sigmoid()
#         self.quantization = quantization        

#     def forward(self, x):
#         if self.quantization:
#             out = self.dequantize(x)
#         else:
#             out = x
#         out = out.view(-1, int(self.feedback_bits / self.B))
#         out = self.fc(out)
#         out = out.view(-1, 2, 126, 128)
#         out = self.decoder_feature(out)
#         out = self.out_cov(out)
#         out = self.sig(out)
#         return out
ACT = nn.GELU()
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
# Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = ACT

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                              bias=True, padding_mode='circular')
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                            bias=True, padding_mode='circular')

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=(inputs.size(2),inputs.size(3))) #inputs.size(3)
        x = self.down(x)
        x = F.leaky_relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        #print(x.shape)
        x = x.repeat(1, 1, inputs.size(2), inputs.size(3))
        #print(x.shape)
        return inputs * x

    

class Bottleneck(nn.Module):# Standard bottleneck
    def __init__(self,c1,c2,shortcut=True,g=1,e=0.5):
        super(Bottleneck,self).__init__()
        c_= int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))




class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        self.att = SEBlock(c2, c2 // 2)

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.att(self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1)))))


    
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out    
    
    
# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(OrderedDict([
#             ("conv3x3_bn_0", ConvBN(2, 64, 3)),
#             ("relu0", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("mean1",nn.AvgPool2d(2, stride=2)),     #8*16
#             ("conv3x3_bn_1", ConvBN(64, 128, 3)),
#              ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("mean2",nn.AvgPool2d(2, stride=2)),     #4*8
#             ("conv3x3_bn_2", ConvBN(128, 256, 3)),
#              ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("mean3",nn.AvgPool2d(2, stride=2)),       #2*4
#              ("conv3x3_bn_3", ConvBN(256, 512, 3)),
#             ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("mean4",nn.AvgPool2d(2, stride=2)), 
#              ("conv3x3_bn_4", ConvBN(512, 256, 3)),
#             ("relu4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("mean5",nn.AvgPool2d(2, stride=2)), 
#             ("conv3x3_bn_5", ConvBN(256, 128, 3)),
#             ("relu5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #              ("mean6",nn.AvgPool2d(2, stride=2)), 
#             ("conv3x3_bn_6", ConvBN(128, 64, 3)),
#             ("relu6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             #("mean7",nn.AvgPool2d(2, stride=2)), 
#                   #1*2
#              ("conv3x3_bn_7", ConvBN(64, 16, 3)),
#             #("relu4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#         ]))
#         self.fc_mu = nn.Linear(896, 128)
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 

#     def forward(self, x):
#         out = self.encoder(x)
#         #print(out.shape)
#         out = out.view(-1,896)
#         #print(out.shape)
#         out = self.fc_mu(out)
#         out = self.sig(out)
#         #print(out.shape)
#         if self.quantization:
#             out = self.quantize(out)
#         else:
#             out = out
#         return out    

# class Interpolate(nn.Module):
#     def __init__(self, scale=2, mode='bilinear'):
#         super(Interpolate, self).__init__()
#         self.interp = nn.functional.interpolate
#         self.scale = scale
#         self.mode = mode
        
#     def forward(self, x):
#         x = self.interp(x,  scale_factor=self.scale, mode=self.mode, align_corners=True)
#         return x    
    
# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         latent_dim=128
        
#         self.decoder_input = nn.Linear(int(self.feedback_bits / self.B), 32256)
        
#         decoder = OrderedDict([
#             ("upsample1",Interpolate()),    #8x8
#             ("conv3x3_bn_1", ConvBN(2, 16, 3)),
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("upsample2",Interpolate()),    #16x16
#             ("conv3x3_bn_2", ConvBN(16, 32, 3)),
#             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("upsample3",Interpolate()),    #32x32
#             ("conv3x3_bn_3", ConvBN(32, 64, 3)),
#             ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
# #             ("upsample4",Interpolate()),    #64x64
#             ("conv3x3_bn_4", ConvBN(64, 128,3)),
#             ("relu4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("upsample5",Interpolate()),   #128x64
#             ("CRBlock1", CRBlock()),
#             ("CRBlock2", CRBlock()),
            
# #             ("conv3x3_bn_5", ConvBN(512, 256,3)),
# #             ("relu5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#              ("conv3x3_bn_6", ConvBN(128, 64,3)),
#             ("relu6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            
#             ("conv3x3_bn_7", ConvBN(64, 32,3)),
#             ("relu7", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#           ])
#         self.decoder_input = nn.Sequential(decoder)
#         self.out_cov = conv3x3(32, 2)
#         self.sig = nn.Sigmoid()
#         self.quantization = quantization        
    
    
    
#     def forward(self, x):
#         if self.quantization:
#             out = self.dequantize(x) 
#         else:
#             out = x
# #         out = out.view(-1, int(self.feedback_bits / self.B))
#         out = out.view(-1, 2, 8, 8)
# #         print(out.shape)
#         result = self.decoder_input(out)
#         #result = result.view(-1,2, 126, 128)

#         #result = self.decoder(result)
#         out = self.out_cov(result)
# #         out = self.final_layer(result)
#         out = self.sig(out[:,:,:126,:])
#         #print(out.shape)
#         return out
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         latent_dim = 128
#         self.feedback_bits = feedback_bits
        
        
#         modules = []
#         hidden_dims = [32,128,32,2]#
#         in_channels=2
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 1, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
# #                     LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
#                     nn.GELU())
#                 #Bottleneck(in_channels,h_dim)
#                 #Bottle2neck(in_channels,h_dim)
#             )
#             in_channels = h_dim
# #         for i in range(3):
# #             modules.append(Bottleneck(128,128))
# #         modules.append(
# #                 nn.Sequential(
# #                     nn.Conv2d(128, 2,
# #                               kernel_size= 3, stride= 1, padding  = 1),
# #                     nn.BatchNorm2d(2),
# #                     nn.ReLU())
# #             )
#         self.encoder1 = nn.Sequential(*modules)
        
        
        
#         modules = []
#         hidden_dims = [16,32,64,64,32,2]
#         in_channels=2
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 1, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
#                    # LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
#                     nn.GELU())
#                 #Bottleneck(in_channels,h_dim)
#                 #Bottle2neck(in_channels,h_dim)
#             )
#             in_channels = h_dim
#         self.encoder2 = nn.Sequential(*modules)
        
        
# #         modules = []
# #         hidden_dims = [16,32,128,32,2]
# #         in_channels=2
# #         # Build Encoder
# #         for h_dim in hidden_dims:
# #             modules.append(
# #                 nn.Sequential(
# #                     nn.Conv2d(in_channels, out_channels=h_dim,
# #                               kernel_size= 3, stride= 1, padding  = 1),
# #                     nn.BatchNorm2d(h_dim),
# #                    # LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
# #                     nn.GELU())
# #             )
# #             in_channels = h_dim
# #         self.encoder3 = nn.Sequential(*modules)
# #         self.encoder3 = nn.Sequential(OrderedDict([
# #             ("conv3x3_bn", ConvBN(2, 2, 3)),
# #             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
# #             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
# #         ]))
        
#         self.encoder_conv = nn.Sequential(
#             nn.Conv2d(4, 2,
#                               kernel_size= 3, stride= 1, padding  = 1)
#         )
        
#         self.fc_mu = nn.Linear(32256, int(feedback_bits / self.B))
#         self.sa = SpatialGate()
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()
#         self.hsig= hsigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 

#     def forward(self, x):
#         #x = (x-x.min())/x.max()
# #         x[:,0,:,:] = F.normalize(x[:,0,:,:],dim=2)
# #         x[:,1,:,:] = F.normalize(x[:,1,:,:],dim=2)
        
#         #x[:,0,:,:] = (x[:,0,:,:]-0.50008255)/0.014866313
#         #x[:,1,:,:] = (x[:,1,:,:]-0.4999794)/0.014148686
#         result1 = self.encoder1(x)
#         result2 = self.encoder2(x)
# #         result3 = self.encoder3(x)
#         result = torch.cat((result1,result2),dim=1)
#         result =self.encoder_conv(result)
#         #print(result.shape)
        
#         result = torch.flatten(result, start_dim=1)
        
        
#         mu = self.fc_mu(result)
#         #mu = mu.view(-1, int(self.feedback_bits / self.B))
#         out = self.sig(mu)
#         #print(out.shape)
#         if self.quantization:
#             out = self.quantize(out)
#         else:
#             out = out
#         return out


    
    
    
# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         latent_dim=128
        
#         self.decoder_input = nn.Linear(int(self.feedback_bits / self.B), 32256)
#         self.refine = RefineBlock()
#         hidden_dims =[16,64,256,512,1024,512,256,64,2]#[16,64,256,512,256,64,2]#[16,64,128,256,128,64,2]#[16,32,64,128,64,32,2]
#         hidden_dims.reverse()
#         modules = []
        
#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride = 1,
#                                        padding=1,
#                                        ),  #output_padding=1
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     #LayerNorm(hidden_dims[i + 1], eps=1e-6, data_format="channels_first"),
#                     nn.GELU())
#             )
# #             if hidden_dims[i+1]==512:
# #                  modules.append(CRBlock())
#         self.decoder1 = nn.Sequential(*modules)
        
        
        
# #         hidden_dims =[16,64,256,512,256,64,16]#[16,64,256,512,256,64,2]#[16,64,128,256,128,64,2]#[16,32,64,128,64,32,2]
# #         hidden_dims.reverse()
# #         modules = []
        
# #         for i in range(len(hidden_dims) - 1):
# #             modules.append(
# #                 nn.Sequential(
# #                     nn.Conv2d(hidden_dims[i],
# #                                        hidden_dims[i + 1],
# #                                        kernel_size=3,
# #                                        stride = 1,
# #                                        padding=1,
# #                                        ),  #output_padding=1
# #                     nn.BatchNorm2d(hidden_dims[i + 1]),
# #                     #LayerNorm(hidden_dims[i + 1], eps=1e-6, data_format="channels_first"),
# #                     nn.GELU())
# #             )
# # #             if hidden_dims[i+1]==512:
# # #                  modules.append(CRBlock())
# #         self.decoder2 = nn.Sequential(*modules)
        
        
        
# #         self.decoder2 = nn.Sequential(OrderedDict([
# #             ("conv5x5_bn", ConvBN(2, 2, 5)),
# #             ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("CRBlock1", CRBlock()),
# #             ("CRBlock2", CRBlock()),

# #         ]))
# #         self.decoder3 = nn.Sequential(OrderedDict([
# #             ("Refine1", RefineBlock()),
# #             ("Refine2", RefineBlock()),
# #         ]))
        
#         self.final_layer = nn.Sequential(
#                             nn.Conv2d(hidden_dims[-1],
#                                                hidden_dims[-1],
#                                                kernel_size=3,
#                                                stride=1,
#                                                padding=1,
#                                                ), #output_padding=1
#                             nn.BatchNorm2d(hidden_dims[-1]),
#                             #LayerNorm(hidden_dims[-1], eps=1e-6, data_format="channels_first"),
#                             nn.GELU(),
#                             nn.Conv2d(hidden_dims[-1],
#                                       out_channels= 2,
#                                       kernel_size= 3, padding= 1),
#                             )

#         self.sig = nn.Sigmoid()
#         self.hsig= hsigmoid()
#         self.quantization = quantization        
    
    
    
#     def forward(self, x):
#         if self.quantization:
#             out = self.dequantize(x) 
#         else:
#             out = x
#         out = out.view(-1, int(self.feedback_bits / self.B))
#         #print(out.shape)
#         result = self.decoder_input(out)
#         result = result.view(-1,2, 126, 128)
#         #result =self.refine(result)
#         allout = []
# #         for i in range(4):
            
# #             result1 = self.decoder1(result[i*16:(i+1)*16])
# #             result2 = self.decoder2(result[i*16:(i+1)*16])
# #             out = torch.cat([result1,result2],1)
# #             out = self.final_layer(out)
# #             out = self.hsig(out)
# #             allout.append(out)
# #         out = torch.cat(allout,0)
#         result = self.decoder1(result)
# #         result2 = self.decoder2(result)
# #         result2 = self.decoder2(result)
# #         result3 = self.decoder3(result)
# #         result = torch.cat([result1,result2],1)
#         out = self.final_layer(result)
#         out = self.hsig(out)
#         #out[:,0,:,:] = out[:,0,:,:]*0.014866313+0.50008255
#         #out[:,1,:,:] = out[:,1,:,:]*0.014148686+0.4999794
        
        
#         #allout.append(out)
#         return out


















# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         latent_dim = 128
#         self.feedback_bits = feedback_bits
      
        
#         modules = []
#         hidden_dims = [32,128,32,2]#
#         in_channels=2
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 1, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
# #                     LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
#                     nn.GELU()
# #                      nn.LeakyReLU(negative_slope=0.3, inplace=True),
#                 #Bottleneck(in_channels,h_dim)
#                 #Bottle2neck(in_channels,h_dim)
#                 )    
                
#             )
#             in_channels = h_dim
# #         for i in range(3):
# #             modules.append(Bottleneck(128,128))
# #         modules.append(
# #                 nn.Sequential(
# #                     nn.Conv2d(128, 2,
# #                               kernel_size= 3, stride= 1, padding  = 1),
# #                     nn.BatchNorm2d(2),
# #                     nn.ReLU())
# #             )
#         self.encoder1 = nn.Sequential(*modules)
        
        
        
#         modules = []
#         hidden_dims = [16,32,64,64,32,2]
#         in_channels=2
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 1, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
#                    # LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
#                     nn.GELU()
# #                      nn.LeakyReLU(negative_slope=0.3, inplace=True),
#                 #Bottleneck(in_channels,h_dim)
#                 #Bottle2neck(in_channels,h_dim)
#                 )
#             )
#             in_channels = h_dim
#         self.encoder2 = nn.Sequential(*modules)
        
        
#         modules = []
#         hidden_dims = [16,32,256,32,16,2]
#         in_channels=2
#         # Build Encoder
#         for h_dim in hidden_dims:
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channels, out_channels=h_dim,
#                               kernel_size= 3, stride= 1, padding  = 1),
#                     nn.BatchNorm2d(h_dim),
#                    # LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
#                     nn.GELU()
# #                      nn.LeakyReLU(negative_slope=0.3, inplace=True)
#                 )
#             )
#             in_channels = h_dim
#         self.encoder3 = nn.Sequential(*modules)
# #         self.encoder3 = nn.Sequential(OrderedDict([
# #             ("conv3x3_bn", ConvBN(2, 2, 3)),
# #             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
# #             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
# #         ]))
        
#         self.encoder_conv = nn.Sequential(
#             nn.Conv2d(4, 2,
#                               kernel_size= 3, stride= 1, padding  = 1)
#         )
        
#         self.fc_mu1 = nn.Linear(15360,96)#int(feedback_bits / self.B))  #32256
#         self.fc_mu2 = nn.Linear(16896,32)#int(feedback_bits / self.B))
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()
#         self.hsig= hsigmoid()
        
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 

#     def forward(self, x0):
#         #x = (x-x.min())/x.max()

# #         x = x.permute(2,1,0,3)
# #         x = x[self.std_sort]
# #         x = x.permute(2,1,0,3)
#         #out=x[:,:][self.std_sort]
#         #print(out.device)
#         x =x0[:,:,:60,:]
#         x = FF.normalize(x,[0.50008255,0.4999794],[0.014866313,0.014148686])
# #         x = (x-0.5000282)/0.014512091
# #         x = FF.normalize(x,[0.5001001,0.49990672],[0.020705294,0.03563241])

#         result1 = self.encoder1(x)
#         result2 = self.encoder2(x)
# #         result3 = self.encoder3(x)
#         result = torch.cat((result1,result2),dim=1)
#         result =self.encoder_conv(result)
#         result = torch.flatten(result, start_dim=1)
        
#         mu = self.fc_mu1(result)
#         out1 = self.sig(mu)
        
        
#         x =x0[:,:,60:,:]
#         x = FF.normalize(x,[0.50008255,0.4999794],[0.014866313,0.014148686])
#         result1 = self.encoder1(x)
#         result2 = self.encoder2(x)
# #         result3 = self.encoder3(x)
#         result = torch.cat((result1,result2),dim=1)
#         result =self.encoder_conv(result)
#         result = torch.flatten(result, start_dim=1)
#         mu = self.fc_mu2(result)
#         out2 = self.sig(mu)
  
        
#         if self.quantization:
#             out1 = self.quantize(out1)
#             out2 = self.quantize(out2)
#             out = torch.cat((out1,out2),1)
# #             out = VQQuant(out,self.codebook)
#         else:
#             out = out
#         return out


    
    
    
# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         latent_dim=128
 
#         self.decoder_input1 = nn.Linear(96, 15360)  #int(self.feedback_bits / self.B)
#         self.decoder_input2 = nn.Linear(32, 16896)  
#         self.refine = RefineBlock()
#         hidden_dims =[16,64,256,512,1024,512,256,64,2]#[16,64,256,512,256,64,2]#[16,64,128,256,128,64,2]#[16,32,64,128,64,32,2]
#         hidden_dims.reverse()
#         modules = []
#         for i in range(len(hidden_dims) - 1):
#             modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(hidden_dims[i],
#                                        hidden_dims[i + 1],
#                                        kernel_size=3,
#                                        stride = 1,
#                                        padding=1,
#                                        ),  #output_padding=1
#                     nn.BatchNorm2d(hidden_dims[i + 1]),
#                     #LayerNorm(hidden_dims[i + 1], eps=1e-6, data_format="channels_first"),
#                     nn.GELU()
# #                     nn.LeakyReLU(negative_slope=0.3, inplace=True),
#                 )
#             )
# #             if hidden_dims[i+1]==512:
# #                  modules.append(CRBlock())
#         self.decoder1 = nn.Sequential(*modules)
        
#         self.decoder2 = nn.Sequential(OrderedDict([
#             ("conv5x5_bn", ConvBN(2, 2, 5)),
#            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #              ("relu", nn.GELU()),
#             ("CRBlock1", CRBlock()),
#             ("CRBlock2", CRBlock()),
# #             ("CRBlock3", CRBlock()),

#         ]))
# #         self.decoder3 = nn.Sequential(OrderedDict([
# #             ("Refine1", RefineBlock()),
# #             ("Refine2", RefineBlock()),
# #         ]))
        
#         self.final_layer = nn.Sequential(
#                             nn.Conv2d(hidden_dims[-1]+2,
#                                                hidden_dims[-1]+2,
#                                                kernel_size=3,
#                                                stride=1,
#                                                padding=1,
#                                                ), #output_padding=1
#                             nn.BatchNorm2d(hidden_dims[-1]+2),
#                             #LayerNorm(hidden_dims[-1], eps=1e-6, data_format="channels_first"),
#                             nn.GELU(),
# #                             nn.LeakyReLU(negative_slope=0.3, inplace=True),
#                             nn.Conv2d(hidden_dims[-1]+2,
#                                       out_channels= 2,
#                                       kernel_size= 3, padding= 1),
#                             )

#         self.sig = nn.Sigmoid()
#         self.hsig= hsigmoid()
#         self.relu = nn.ReLU()
#         self.quantization = quantization        
# #         for m in self.modules():
# #             if isinstance(m, (nn.Conv2d, nn.Linear)):
# #                 nn.init.xavier_uniform_(m.weight)
# #             elif isinstance(m, nn.BatchNorm2d):
# #                 nn.init.constant_(m.weight, 1)
# #                 nn.init.constant_(m.bias, 0)
    
    
#     def forward(self, x):
#         if self.quantization:
#             out1 = self.dequantize(x[:,:384])
#             out2 = self.dequantize(x[:,384:])
#             #print(out1.shape,out2.shape)
# #             out = VQDeQuant(x,self.codebook)
#         else:
#             out = x
#         #print(out.shape)
#         out = out1.view(-1,96) #int(self.feedback_bits / self.B)
#         result = self.decoder_input1(out)
#         result = result.view(-1,2, 60, 128)
#         result1 = self.decoder1(result)
#         result2 = self.decoder2(result)
# #         result3 = self.decoder3(result)
#         out = torch.cat([result1,result2],1)
#         out1 = self.final_layer(out)
        
        
        
#         out = out2.view(-1,32) #int(self.feedback_bits / self.B)
#         result = self.decoder_input2(out)
#         result = result.view(-1,2, 66, 128)
#         result1 = self.decoder1(result)
#         result2 = self.decoder2(result)
# #         result3 = self.decoder3(result)
#         out = torch.cat([result1,result2],1)
#         out2 = self.final_layer(out)
        
#         #elsePart2 = torch.ones([out.shape[0],2,66,128],device=out.device)*0.5
#         #print(out1.shape,out2.shape)
# #         out = FF.normalize(out,[-0.5001001/0.020705294,-0.49990672/0.03563241],[1/0.020705294,1/0.03563241])
# #         out = out*0.014512091+0.5000282
#         out = torch.cat((out1,out2),2)
#         out = FF.normalize(out,[-0.50008255/0.014866313,-0.4999794/0.014148686],[1/0.014866313,1/0.014148686])
# #         out = torch.cat((out,elsePart2),2)
#         return out

class ACRDecoderBlock(nn.Module):
    r""" Inverted residual with extensible width and group conv
    """
    def __init__(self, expansion):
        super(ACRDecoderBlock, self).__init__()
        width = 8 * expansion
        self.conv1_bn = ConvBN(2, width, [1, 9])
        self.prelu1 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv2_bn = ConvBN(width, width, 7, groups=4 * expansion)
        self.prelu2 = nn.PReLU(num_parameters=width, init=0.3)
        self.conv3_bn = ConvBN(width, 2, [9, 1])
        self.prelu3 = nn.PReLU(num_parameters=2, init=0.3)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)

        residual = self.prelu1(self.conv1_bn(x))
        residual = self.prelu2(self.conv2_bn(residual))
        residual = self.conv3_bn(residual)

        return self.prelu3(identity + residual)


class Fire(nn.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int):
        super().__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1
        )






class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        latent_dim = 128
        self.feedback_bits = feedback_bits
        self.std_sort= [ 31,  32,  27,  28,  33,  30,  34,  35,  29,  36,  26,  37,  38,
        39,  40,  25,  41,  42,  43,  44,  45,  46,  24,  47,  48,  49,
        50,  51,  52,  53,  54,  55,  56,  20,  23,  58,  59,  22,  21,
        57,  61,  60,  64,  62,  19,  63,  68,  65,  69,  67,  70,  66,
        73,  77,  78,  71,  79,  84,  86,  80,  72,  76,  88,  18,  74,
        85,  75,  83, 101,  87,  89,  82, 102,  81,  13,  15,  17,  90,
        10,  11,  91,  92, 106,   8,  93, 107,  12, 100, 111,  95, 104,
       103,  14,  16,  94,   3,  99,   9, 120, 105,   0,   7,   6,   1,
       116,  98,  96,   4,  97, 110,   5, 119, 115,   2, 108, 112, 117,
       113, 114, 125, 121, 124, 109, 118, 123, 122]

        modules = []
        hidden_dims = [32,128,32,2]#
        in_channels=2
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1,bias=False),#,bias=False
                    nn.BatchNorm2d(h_dim),
#                     LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
                    nn.GELU()
#                     nn.PReLU(num_parameters=h_dim, init=0.3),
                )
                #Bottleneck(in_channels,h_dim)
                #Bottle2neck(in_channels,h_dim)
            )
            in_channels = h_dim
#         for i in range(3):
#             modules.append(Bottleneck(128,128))
#         modules.append(
#                 nn.Sequential(
#                     nn.Conv2d(128, 2,
#                               kernel_size= 3, stride= 1, padding  = 1),
#                     nn.BatchNorm2d(2),
#                     nn.ReLU())
#             )
        self.encoder1 = nn.Sequential(*modules)
        
        
        
        modules = []
        hidden_dims =[16,32,64,64,32,2]
        in_channels=2
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1,bias=False),
                    nn.BatchNorm2d(h_dim),
#                     nn.InstanceNorm2d(h_dim),
                   # LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
#                     nn.PReLU(num_parameters=h_dim, init=0.3),
                    #Fire(h_dim, 16, h_dim//2, h_dim//2),
                
                )
                #Bottleneck(in_channels,h_dim)
                #Bottle2neck(in_channels,h_dim)
            )
            in_channels = h_dim
        self.encoder2 = nn.Sequential(*modules)
        
        
        modules = []
        hidden_dims = [16,32,256,32,16,2]
        in_channels=2
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    nn.BatchNorm2d(h_dim),
#                     nn.InstanceNorm2d(h_dim),
                   # LayerNorm(h_dim, eps=1e-6, data_format="channels_first"),
                    nn.GELU())
            )
            in_channels = h_dim
#         self.encoder3 = nn.Sequential(*modules)
#         self.encoder3 = nn.Sequential(OrderedDict([
#             ("conv3x3_bn", ConvBN(2, 2, 3)),
#             ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
#             ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
#         ]))
#         self.encoder3 =ConvBN(in_channels, 2,1)
#         self.encoder3 = nn.Sequential(OrderedDict([
#             ("conv5x5_bn", ConvBN(2, 2, 5)),
#             ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ("CRBlock1", CRBlock()),
#             ("CRBlock2", CRBlock()),
# #              ("CRBlock3", CRBlock()),
#         ]))
#         self.encoder3 =ConvBN(2, 2,1)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(4, 2,
                              kernel_size= 3, stride= 1, padding  = 1),
#             nn.BatchNorm2d(2),
#             nn.InstanceNorm2d(2),
        )
        
        self.fc_mu = nn.Linear(15360, int(feedback_bits / self.B))  #32256  #
#         self.fc_mu2 = nn.Linear(2048, int(feedback_bits / self.B)) 
        #self.norm = nn.BatchNorm1d(2048)
        self.sa = SpatialGate()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.hsig= hsigmoid()
        self.drop = nn.Dropout(0.1)
#         origin_codebook = (torch.arange(2**2).float()+0.5)/2**2
#         origin_codebook = origin_codebook.view(1,1,1,-1).repeat(1,256,1,1)
#         self.codebook = nn.Parameter(origin_codebook)
        
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 

    def forward(self, x):
        #x = (x-x.min())/x.max()
#         x = (x-0.5000282)/0.014512091
#         x = x.permute(2,1,0,3)
#         x = x[self.std_sort]
#         x = x.permute(2,1,0,3)
        #out=x[:,:][self.std_sort]
        #print(out.device)
        
        x =x[:,:,:60,:]
        x = FF.normalize(x,[0.50008255,0.4999794],[0.014866313,0.014148686])

#         x = (x-x.mean(0))/x.std(0)
#         x = FF.normalize(x,[0.5001001,0.49990672],[0.020705294,0.03563241])

#         x[:,0,:,:]=(x[:,0,:,:]- torch.Tensor(0.50008255))/0.014866313
#         x[:,1,:,:]=(x[:,1,:,:]- 0.4999794)/0.014148686
        result1 = self.encoder1(x)
        result2 = self.encoder2(x)
#         result3 = self.encoder3(x)
#         result3 = self.encoder3(x)
        result = torch.cat((result1,result2),dim=1)
#         result = result1+result2
        result =self.encoder_conv(result)
        #print(result.shape)
        
        result = torch.flatten(result, start_dim=1)
        
        
        mu = self.fc_mu(result)
        #mu = self.norm(mu)
        #mu = self.fc_mu2(mu)
        #mu = mu.view(-1, int(self.feedback_bits / self.B))
        out = self.sig(mu)
        #print(out.shape)
        if self.quantization:
            out = self.quantize(out)
            #out = VQQuant(out.view(out.shape[0],-1,1),self.codebook)
            #print(out.shape)
        else:
            out = out
        return out


    
    
    
class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        latent_dim=128
        
        self.std_sort=[100, 103, 113,  95, 107, 110, 102, 101,  83,  97,  78,  79,  86,
        74,  92,  75,  93,  76,  63,  44,  33,  38,  37,  34,  22,  15,
        10,   2,   3,   8,   5,   0,   1,   4,   6,   7,   9,  11,  12,
        13,  14,  16,  17,  18,  19,  20,  21,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  39,  35,  36,  41,  40,  43,  45,  42,
        47,  51,  49,  46,  48,  50,  55,  60,  52,  64,  66,  61,  53,
        54,  56,  59,  73,  71,  67,  57,  65,  58,  69,  62,  70,  77,
        80,  81,  84,  94,  89, 106, 108, 105,  96,  87,  68,  72,  91,
        90,  99,  82,  85, 114, 122, 109,  88, 115, 117, 118, 112, 104,
       116, 123, 111,  98, 120, 125, 124, 121, 119]
#         origin_codebook = (torch.arange(2**2).float()+0.5)/2**2
#         origin_codebook = origin_codebook.view(1,1,1,-1).repeat(1,256,1,1)
#         self.codebook = nn.Parameter(origin_codebook)
        
        self.decoder_input = nn.Linear(int(self.feedback_bits / self.B), 15360)   #
    
        
        self.refine = RefineBlock()
        hidden_dims =[32,128,256,512,1024,512,256,128,2]#[16,64,256,512,256,64,2]#[16,64,128,256,128,64,2]#[16,32,64,128,64,32,2]
        hidden_dims.reverse()
        modules = []
#         modules.append(nn.Sequential(OrderedDict([
#             ("conv5x5_bn", ConvBN(2, 2, 5)),
#             ("relu", nn.GELU()),])))
        for i in range(len(hidden_dims) - 1):
              
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(hidden_dims[i],
                                           hidden_dims[i + 1],
                                           kernel_size=3,
                                           stride = 1,
                                           padding=1,
                                  
                                           ),  #output_padding=1
                        nn.BatchNorm2d(hidden_dims[i + 1]),
#                         LayerNorm(hidden_dims[i + 1], eps=1e-6, data_format="channels_first"),
                        nn.GELU(),
#                         SELayer(hidden_dims[i + 1])
    #                     nn.PReLU(num_parameters=hidden_dims[i + 1], init=0.3), 
                    )
                )
#             if hidden_dims[i+1]==512:
#                  modules.append(CRBlock())
        self.decoder1 = nn.Sequential(*modules)
        

      
        self.decoder2 = nn.Sequential(OrderedDict([
            ("conv5x5_bn", ConvBN(2, 16, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock()),
            ("CRBlock2", CRBlock()),
#              ("CRBlock3", CRBlock()),
        ]))
        
#         self.decoder3 = nn.Sequential(OrderedDict([
#             ("conv5x5_bn", ConvBN(2, 2, 5)),
#             ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
# #             ("CRBlock1", CRBlock()),
# #             ("CRBlock2", CRBlock()),
#               ("CRBlock1", ACRDecoderBlock(expansion=16)),
#             ("CRBlock2", ACRDecoderBlock(expansion=16)),
            
#         ]))


#         hidden_dims =[16,128,256,512,1024,512,256,128,2]#[16,64,256,512,256,64,2]#[16,64,128,256,128,64,2]#[16,32,64,128,64,32,2]
#         hidden_dims.reverse()
#         modules = []
# #         modules.append(nn.Sequential(OrderedDict([
# #             ("conv5x5_bn", ConvBN(2, 2, 5)),
# #             ("relu", nn.GELU()),])))
#         for i in range(len(hidden_dims) - 1):
              
#                 modules.append(
#                     nn.Sequential(
#                         nn.Conv2d(hidden_dims[i],
#                                            hidden_dims[i + 1],
#                                            kernel_size=3,
#                                            stride = 1,
#                                            padding=1,
                                  
#                                            ),  #output_padding=1
#                         nn.BatchNorm2d(hidden_dims[i + 1]),
# #                         LayerNorm(hidden_dims[i + 1], eps=1e-6, data_format="channels_first"),
#                         nn.GELU(),
#                     )
#                 )

        self.decoder3 = nn.Sequential(*modules)
        self.in_cov = ConvBN(2, 2,5)
        self.multiConvs = nn.ModuleList()
        for i in range(5):
            self.multiConvs.append(nn.Sequential(
                conv3x3_bn(2, (i+1)*16),
#                 nn.ReLU(),
                nn.GELU(),
                conv3x3_bn((i+1)*16, (i+1)*32),
#                 nn.ReLU(),
                nn.GELU(),
                conv3x3_bn((i+1)*32, (i+1)*16),
#                 nn.ReLU(),
                nn.GELU(),
                conv3x3_bn((i+1)*16, 2),
#                 nn.ReLU()
                nn.GELU(),
            ))

        
        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_dims[-1]+16,
                                               hidden_dims[-1]+16,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                            
                                               ), #output_padding=1
                            nn.BatchNorm2d(hidden_dims[-1]+16),
                            #LayerNorm(hidden_dims[-1], eps=1e-6, data_format="channels_first"),
#                             nn.PReLU(num_parameters=hidden_dims[-1]+2, init=0.3),
                            nn.GELU(),
                            nn.Conv2d(hidden_dims[-1]+16,
                                      out_channels= 2,
                                      kernel_size= 3, padding= 1,
                                     ),
                            )


        self.sig = nn.Sigmoid()
        self.hsig= hsigmoid()
        self.relu = nn.ReLU()
        self.quantization = quantization        
    
    
    
    def forward(self, x):
        if self.quantization:
            out = self.dequantize(x) 
#             out = VQDeQuant(x,self.codebook)
        else:
            out = x
        out = out.view(-1, int(self.feedback_bits / self.B)) #int(self.feedback_bits / self.B)
        #print(out.shape)
        #out = out.half()
        result = self.decoder_input(out)
        result = result.view(-1,2, 60, 128)
        #result =self.refine(result)
#         allout = []
#         for i in range(4):
            
#             result1 = self.decoder1(result[i*16:(i+1)*16])
#             result2 = self.decoder2(result[i*16:(i+1)*16])
#             out = torch.cat([result1,result2],1)
#             out = self.final_layer(out)
#             out = self.hsig(out)
#             allout.append(out)
#         out = torch.cat(allout,0)
        

            
        result1 = self.decoder1(result)
        result2 = self.decoder2(result)
        
#         out = self.in_cov(result)
#         for i in range(5):
#             residual = out
#             out = self.multiConvs[i](out)
#             out = residual + out
        
#         result3 = self.decoder3(result)
        #result1= result1+result3
        out = torch.cat([result1,result2],1)
#         out = result1+result2
        out = self.final_layer(out)
        
#         out = self.sig(out)
#         out = out*0.014512091+0.5000282
#         out[:,0,:,:]=out[:,0,:,:]*0.014866313+0.50008255
#         out[:,1,:,:]=out[:,1,:,:]*0.014148686+0.4999794
#          x[:,0]=(x[:,0]- 0.50008255)/0.014866313
#         x[:,1]=(x[:,1]- 0.4999794)/0.014148686
#         result1 = self.decoder1(result)
#         result2 = self.decoder2(result)
#         result = torch.cat([result1,result2],1)
#         out = self.final_layer(result)
#         out = self.hsig(out)
        #allout.append(out)
#         out=out[:,:][torch.Tensor(self.std_sort).long().to(out.device)]

        #elsePart1 = torch.ones([out.shape[0],2,18,128],device=out.device)*0.5
        elsePart2 = torch.ones([out.shape[0],2,66,128],device=out.device)*0.5

        out = FF.normalize(out,[-0.50008255/0.014866313,-0.4999794/0.014148686],[1/0.014866313,1/0.014148686])
#         out = FF.normalize(out,[-0.5001001/0.020705294,-0.49990672/0.03563241],[1/0.020705294,1/0.03563241])
        
        out = torch.cat((out,elsePart2),2)
#         out = out.permute(2,1,0,3)
#         out = out[self.std_sort]
#         out= out.permute(2,1,0,3)
        return out


# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits):
#         super(Encoder, self).__init__()
#         self.conv1 = conv3x3_bn(2, 256)
#         self.conv2 = conv3x3_bn(256, 2)
#         self.fc = nn.Linear(15360, int(feedback_bits // self.B))
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)

#     def forward(self, x):
#         x =x[:,:,:60,:]
#         x = FF.normalize(x,[0.50008255,0.4999794],[0.014866313,0.014148686])
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = out.view(-1, 15360)
#         out = self.fc(out)
#         out = self.sig(out)
#         out = self.quantize(out)

#         return out

# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
        
#         self.fc = nn.Linear(int(feedback_bits // self.B), 15360)
        
#         self.sig = nn.Sigmoid()
# #         self.in_cov = conv3x3_bn(2, 16)
#         self.in_cov = ConvBN(2,16,5)
#         self.multiConvs = nn.ModuleList()
#         for i in range(5):
#             self.multiConvs.append(nn.Sequential(
#                 conv3x3_bn(16, (i+1)*16),
# #                 nn.ReLU(),
#                 nn.GELU(),
#                 conv3x3_bn((i+1)*16, (i+1)*32),
# #                 nn.ReLU(),
#                 nn.GELU(),
#                 conv3x3_bn((i+1)*32, (i+1)*16),
# #                 nn.ReLU(),
#                 nn.GELU(),
#                 conv3x3_bn((i+1)*16, 16),
# #                 nn.ReLU()
#                 nn.GELU(),
#             ))
#         self.out_cov = conv3x3_bn(16, 2)
#     def forward(self, x):
#         out = self.dequantize(x)
#         out = out.view(-1, int(self.feedback_bits // self.B))
#         out = self.sig(self.fc(out))
#         out = out.view(-1,2, 60, 128)
#         out = self.in_cov(out)
#         for i in range(5):
#             residual = out
#             out = self.multiConvs[i](out)
#             out = residual + out

#         out = self.out_cov(out)
# #         out = self.sig(out)
#         elsePart2 = torch.ones([out.shape[0],2,66,128],device=out.device)*0.5

#         out = FF.normalize(out,[-0.50008255/0.014866313,-0.4999794/0.014148686],[1/0.014866313,1/0.014148686])
# #         out = FF.normalize(out,[-0.5001001/0.020705294,-0.49990672/0.03563241],[1/0.020705294,1/0.03563241])
        
#         out = torch.cat((out,elsePart2),2)
#         return out














def positional_encoding(X, num_features, dropout_p=0.1, max_len=512):
    r'''
        给输入加入位置编码
    参数：
        - num_features: 所取得特征向量得维度
        - dropout_p: dropout的概率，当其为非零时执行dropout
        - max_len: ，默认512
    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]
    例子：
        >>> X = torch.randn((2,4,10))
        >>> X = positional_encoding(X, 10)
        >>> print(X.shape)
        >>> torch.Size([2, 4, 10])
    '''
    """
    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)
    """
    num_features_ = num_features
    max_len_ = max_len
    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2, dtype=torch.float32) / num_features)
    P[:, :, 0::2] = torch.sin(X_)
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)
    return X    
    
  


# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         latent_dim = 128
#         self.feedback_bits = feedback_bits
#         #self.embedding = nn.Linear(128, 512)
# #         self.conv = nn.Conv2d(2, out_channels=1,
# #                               kernel_size= 3, stride= 1, padding  = 1)
#         self.pos = nn.Embedding(126,256)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
#         self.tran sformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)  
       
#         self.norm = nn.LayerNorm(256)
#         self.fc_mu = nn.Linear(256, int(feedback_bits / self.B))
        
#         self.relu = nn.ReLU()
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 

#     def forward(self, x):
#         b,c,w,h = x.detach().size()
#         #x = torch.cat((x[:,0,:,:],x[:,1,:,:]),dim=1)
#         #print(x.shape)
#         #x = self.conv(x).squeeze()#.permute(1,0,2)
#         #x = x.transpose(1, 2)
#         x = x.view(-1,126,256)
#         #x = self.embedding(x)
#         #x = positional_encoding(x, x.shape[-1]).permute(1,0,2)
#         x = x + self.pos(torch.arange(0,126).long().cuda())
#         #x = x.reshape(b,126,256).permute(1,0,2)
#         #.permute(1,0,2)
#         #x = self.norm(x)
# #         x = self.transformer_encoder(x.permute(1,0,2)).permute(1,0,2)
#         x=torch.transpose(x, 0, 1)
#         x = self.transformer_encoder(x)#.permute(1,0,2)
#         x=torch.transpose(x, 0, 1)
#         x=x[:, 0]
#         #print(x.shape)
#         result = torch.flatten(x, start_dim=1)
# #         result = result.view(-1, int(self.feedback_bits / self.B))
#         #print(result.shape)
#         mu = self.fc_mu(result)

#         out = self.sig(mu)
#         #print(out.shape)
#         if self.quantization:
#             out = self.quantize(out)
#         else:
#             out = out
#         return out

# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Decoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         latent_dim=128
#         self.norm = nn.LayerNorm(256)
        
#         self.decoder_input = nn.Linear(int(self.feedback_bits / self.B), 256)
#         self.dec_tokens = torch.nn.Parameter(torch.randn((1, 126, 256)))
#         self.pos = nn.Embedding(126,256)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
#         self.transformer_decoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)  
#         self.final_layer = nn.Sequential(
# #                             nn.Conv2d(2,
# #                                                2,
# #                                                kernel_size=3,
# #                                                stride=1,
# #                                                padding=1,
# #                                                ), #output_padding=1
# #                             nn.BatchNorm2d(2),
# #                             #LayerNorm(hidden_dims[-1], eps=1e-6, data_format="channels_first"),
# #                             nn.GELU(),
#                             nn.Conv2d(2,
#                                       out_channels= 2,
#                                       kernel_size= 3, padding= 1),
#                             )
#         self.sig = nn.Sigmoid()
#         self.quantization = quantization        
    
    
    
#     def forward(self, x):
#         if self.quantization:
#             out = self.dequantize(x) 
#         else:
#             out = x
#         out = out.view(-1, int(self.feedback_bits / self.B))
#         #print(out.shape)
        
#         result = self.decoder_input(out)
#         #print(result)
#         result = torch.cat([result[:, None], self.dec_tokens.repeat(result.shape[0], 1, 1).to(result.device)], 1)
#         #result = result.view(-1,252,128)
# #         print(result.shape)
#         #result = positional_encoding(result, result.shape[-1]).permute(1,0,2)
#         result = result + self.pos(torch.range(0,126).long().to(result.device))
#         #result = self.norm(result)
#         result=torch.transpose(result, 0, 1)
        
#         print(result.shape)
#         result = self.transformer_decoder(result)#.permute(1,0,2)
#         result = torch.transpose(result, 0, 1)
#         #out = torch.cat((result[:,:,:128].unsqueeze(1),result[:,:,128:].unsqueeze(1)),dim=1)
# #         out = result.unsqueeze(1)
# #         out =self.final_layer(out)
#         out = result[:, 1:]
# #         out =result.view(-1,2,126,128)
#         #out =self.final_layer(out)
#         out = self.sig(out)
#         out=out.reshape(-1,2, 126, 128,) 
#         return out



# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         self.encoder1 =ConvNeXt(in_chans=2,depths=[1,1, 1, 1], dims=[96, 192, 384, 768])

#         self.fc = nn.Linear(768, int(feedback_bits / self.B))
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 

#     def forward(self, x):
#         out = self.encoder1(x)
        
#         out = out.view(-1, 768)
#         out = self.fc(out)
#         out = self.sig(out)
#         if self.quantization:
#             out = self.quantize(out)
#         else:
#             out = out
#         return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 512 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out

# def Num2Bit(Num, B):
#     Num_ = Num.type(torch.uint8)

#     def integer2bit(integer, num_bits=B * 2):
#         dtype = integer.type()
#         exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
#         exponent_bits = exponent_bits.repeat(integer.shape + (1,))
#         out = integer.unsqueeze(-1) // 2 ** exponent_bits
#         return (out - (out % 1)) % 2

#     bit = integer2bit(Num_)
#     bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
#     return bit.type(torch.float32)


# def Bit2Num(Bit, B):
#     Bit_ = Bit.type(torch.float32)
#     Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
#     num = torch.zeros(Bit_[:, :, 1].shape).cuda()
#     for i in range(B):
#         num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
#     return num


# class Quantization(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, B):
#         ctx.constant = B
#         step = 2 ** B
#         out = torch.round(x * step - 0.5)
#         out = Num2Bit(out, B)
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         # return as many input gradients as there were arguments.
#         # Gradients of constant arguments to forward must be None.
#         # Gradient of a number is the sum of its four bits.
#         b, _ = grad_output.shape
#         grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
#         return grad_num, None


# class Dequantization(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, B):
#         ctx.constant = B
#         step = 2 ** B
#         out = Bit2Num(x, B)
#         out = (out + 0.5) / step
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         # return as many input gradients as there were arguments.
#         # Gradients of non-Tensor arguments to forward must be None.
#         # repeat the gradient of a Num for four time.
#         #b, c = grad_output.shape
#         #grad_bit = grad_output.repeat(1, 1, ctx.constant) 
#         #return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
#         grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
#         return grad_bit, None


# class QuantizationLayer(nn.Module):

#     def __init__(self, B):
#         super(QuantizationLayer, self).__init__()
#         self.B = B

#     def forward(self, x):
#         out = Quantization.apply(x, self.B)
#         return out


# class DequantizationLayer(nn.Module):

#     def __init__(self, B):
#         super(DequantizationLayer, self).__init__()
#         self.B = B

#     def forward(self, x):
#         out = Dequantization.apply(x, self.B)
#         return out


# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=1, bias=True)


# class ConvBN(nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
#         if not isinstance(kernel_size, int):
#             padding = [(i - 1) // 2 for i in kernel_size]
#         else:
#             padding = (kernel_size - 1) // 2
#         super(ConvBN, self).__init__(OrderedDict([
#             ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
#                                padding=padding, groups=groups, bias=False)),
#             ('bn', nn.BatchNorm2d(out_planes))
#         ]))


# class CRBlock(nn.Module):
#     def __init__(self):
#         super(CRBlock, self).__init__()
#         self.path1 = nn.Sequential(OrderedDict([
#             ('conv3x3', ConvBN(256, 256, 3)),
#             ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('conv1x9', ConvBN(256, 256, [1, 9])),
#             ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('conv9x1', ConvBN(256, 256, [9, 1])),
#         ]))
#         self.path2 = nn.Sequential(OrderedDict([
#             ('conv1x5', ConvBN(256, 256, [1, 5])),
#             ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
#             ('conv5x1', ConvBN(256, 256, [5, 1])),
#         ]))
#         self.conv1x1 = ConvBN(256 * 2, 256, 1)
#         self.identity = nn.Identity()
#         self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

#     def forward(self, x):
#         identity = self.identity(x)

#         out1 = self.path1(x)
#         out2 = self.path2(x)
#         out = torch.cat((out1, out2), dim=1)
#         out = self.relu(out)
#         out = self.conv1x1(out)

#         out = self.relu(out + identity)
#         return out





def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse


def NMSE_cuda(x, x_hat):
    x_real = x[:, :, :, 0].view(len(x),-1) - 0.5
    x_imag = x[:, :, :, 1].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, :, :, 0].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, :, :, 1].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
    nmse = mse/power
    return nmse

class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse


def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __getitem__(self, index):
        return self.matdata[index]

    def __len__(self):
        return self.matdata.shape[0]


    
class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])
        self.att = SEBlock(c2, c2 // 2)

    def forward(self, x):
        return self.att(self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, c1, c2, gain=2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.gain = gain
        self.conv = Conv(c1 // 4, c2, k, s, p, g, act)

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return self.conv(x.view(N, C // s ** 2, H * s, W * s))  # x(1,16,160,160)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class WLBlock(nn.Module):
    def __init__(self, paths, in_c, k=16, n=[1, 1], e=[1.0, 1.0], quantization=True):

        super(WLBlock, self).__init__()
        self.paths = paths
        self.n = n
        self.e = e
        self.k = k
        self.in_c = in_c
        for i in range(self.paths):
            self.__setattr__(str(i), nn.Sequential(OrderedDict([
                ("Conv0", Conv(self.in_c, self.k, 3)),
                ("BCSP_1", BottleneckCSP(self.k, self.k, n=self.n[i], e=self.e[i])),
                ("C3_1", C3(self.k, self.k, n=self.n[i], e=self.n[i])),
                ("Conv1", Conv(self.k, self.k, 3)),
            ])))
        self.conv1 = conv3x3(self.k * self.paths, self.k)

    def forward(self, x):
        outs = []
        for i in range(self.paths):
            _ = self.__getattr__(str(i))(x)
            outs.append(_)
        out = torch.cat(tuple(outs), dim=1)
        out = self.conv1(out)
        out = out + x if self.in_c == self.k else out
        return out

    
    
    
# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
#         self.feedback_bits = feedback_bits
#         self.k = 256
#         self.encoder1 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 16, 5)),
#             ("BCSP_1", BottleneckCSP(16, 16, n=2, e=0.5)),
#             ("C3_1", C3(16, 16, n=1, e=2.0)),
#             ("Conv1", Conv(16, self.k, 3))
#         ]))
#         self.encoder2 = nn.Sequential(OrderedDict([
#             ("Focus0", Focus(2, 16)),
#             ("BCSP_1", BottleneckCSP(16, 16, n=1, e=1.0)),
#             ("C3_1", C3(16, 16, n=2, e=2.0)),
#             ("Expand0", Expand(16, 16)),
#             ("Conv1", Conv(16, self.k, 3))
#         ]))
#         self.encoder3 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 32, 3)),
#             ("WLBlock1", WLBlock(3, 32, 32, [1, 2, 3], [0.5, 1, 1.5])),
#             ("WLBlock2", WLBlock(2, 32, 32, [2, 4], [1, 2])),
#             ("Conv1", Conv(32, self.k, 3)),
#         ]))
#         self.encoder_conv = nn.Sequential(OrderedDict([
#             ("conv1x1", Conv(self.k * 3, 2, 1)),
#         ]))
#         #self.fc = nn.Linear(15360, int(15360 / 16))
#         self.dim_verify = nn.Linear(15360, int(self.feedback_bits / self.B))

#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization

#     def forward(self, x):
# #         if channel_last:
# #             x = x.permute(0, 3, 1, 2).contiguous()
#         x =x[:,:,:60,:]
#         x = FF.normalize(x,[0.50008255,0.4999794],[0.014866313,0.014148686])
#         x0 = x.view(-1, 15360)
#         encoder1 = self.encoder1(x)
#         encoder2 = self.encoder2(x)
#         encoder3 = self.encoder3(x)
#         out = torch.cat((encoder1, encoder2, encoder3), dim=1)
#         out = self.encoder_conv(out)
#         out = out.view(-1, 15360) #+ x0
#         #out = self.fc(out)
#         out = self.dim_verify(out)
#         out = self.sig(out)
#         enq_data = out
#         if self.quantization:
#             out = self.quantize(out)
#         elif self.quantization == 'check':
#             out = out
#         else:
#             out = self.fake_quantize(out)
#         return out
# REFINEMENT=0

# class Decoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Decoder, self).__init__()
#         self.k = 16
#         self.feedback_bits = feedback_bits
#         self.dequantize = DequantizationLayer(self.B)
#         #self.dim_verify = nn.Linear(int(self.feedback_bits / self.B), int(15360 / 16))
#         self.fc = nn.Linear(int(self.feedback_bits / self.B), 15360)
#         self.ende_refinement = nn.Sequential(
#             nn.Linear(int(self.feedback_bits / self.B), int(self.feedback_bits / self.B)),
#             nn.BatchNorm1d(int(self.feedback_bits / self.B)),
#             nn.ReLU(True),
#             nn.Linear(int(self.feedback_bits / self.B), int(self.feedback_bits / self.B), bias=False),
#             nn.Sigmoid(),
#         )
#         self.decoder1 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 16, 3)),
#             ("BCSP_1", BottleneckCSP(16, 16, n=1, e=1.0)),
#             ("Conv1", Conv(16, self.k, 1)),
#         ]))
#         self.decoder2 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 32, 5)),
#             ("BCSP_1", BottleneckCSP(32, 32, n=4, e=2.0)),
#             ("Conv1", Conv(32, self.k, 1)),
#         ]))
#         self.decoder3 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 32, (1, 3))),
#             ("BCSP_1", BottleneckCSP(32, 32, n=4, e=2.0)),
#             ("Conv1", Conv(32, self.k, 1)),
#         ]))
#         self.decoder4 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 32, (3, 1))),
#             ("BCSP_1", BottleneckCSP(32, 32, n=4, e=2.0)),
#             ("Conv1", Conv(32, self.k, 1)),
#         ]))
#         self.decoder5 = nn.Sequential(OrderedDict([
#             ("Focus0", Focus(2, self.k)),
#             ("WLBlock1", WLBlock(3, self.k, self.k, [1, 2, 3], [0.5, 1, 1.5])),
#             ("WLBlock2", WLBlock(2, self.k, self.k, [2, 4], [1, 2])),
#             ("Expand0", Expand(self.k, self.k)),
#             ("Conv1", Conv(self.k, self.k, 1)),
#         ]))
#         self.decoder6 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 32, (3, 5))),
#             ("BCSP_1", BottleneckCSP(32, 32, n=4, e=2.0)),
#             ("Conv1", Conv(32, self.k, 5)),
#         ]))
#         self.decoder7 = nn.Sequential(OrderedDict([
#             ("Conv0", Conv(2, 32, (5, 3))),
#             ("BCSP_1", BottleneckCSP(32, 32, n=4, e=2.0)),
#             ("Conv1", Conv(32, self.k, 3)),
#         ]))
#         self.decoder8 = nn.Sequential(OrderedDict([
#             ("Focus0", Focus(2, self.k, 5)),
#             ("WLBlock1", WLBlock(2, self.k, self.k, [1, 2], [0.5, 1])),
#             ("WLBlock2", WLBlock(2, self.k, self.k, [1, 2], [1, 0.5])),
#             ("Expand0", Expand(self.k, self.k)),
#             ("Conv1", Conv(self.k, self.k, 5)),
#         ]))
#         if REFINEMENT:
#             self.refinemodel = nn.Sequential(OrderedDict([
#                 ("Conv0", Conv(2, 64, 3)),
#                 ("WLBlock1", WLBlock(3, 64, 64, [1, 2, 3], [0.5, 1, 1.5])),
#                 ("WLBlock2", WLBlock(2, 64, 64, [2, 4], [1, 2])),
#                 ("WLBlock3", WLBlock(2, 64, 64, [2, 4], [1, 2])),
#                 ("WLBlock4", WLBlock(2, 64, 64, [1, 3], [1, 2])),
#                 ("Conv1", Conv(64, 2, 3)),
#             ]))
#         self.decoder_conv = conv3x3(self.k * 8, 2)
#         self.sig = nn.Sigmoid()
#         self.quantization = quantization

#     def forward(self, x):
#         if self.quantization:
#             out = self.dequantize(x)
#         else:
#             out = x
#         out = out.view(-1, int(self.feedback_bits / self.B))
#         out_error = self.ende_refinement(out)
#         out = out + out_error - 0.5
#         deq_data = out
#         #out = self.dim_verify(out)

#         out = self.sig(self.fc(out))
#         out = out.view(-1, 2, 60, 128)
#         out0 = out
#         out1 = self.decoder1(out)
#         out2 = self.decoder2(out)
#         out3 = self.decoder3(out)
#         out4 = self.decoder4(out)
#         out5 = self.decoder5(out)
#         out6 = self.decoder6(out)
#         out7 = self.decoder7(out)
#         out8 = self.decoder8(out)
#         out = torch.cat((out1, out2, out3, out4, out5, out6, out7, out8), dim=1)
#         out = self.decoder_conv(out) #+ out0
# #         out = self.sig(out)
#         if REFINEMENT:
#             out = self.sig(self.refinemodel(out)) - 0.5 + out
#         elsePart2 = torch.ones([out.shape[0],2,66,128],device=out.device)*0.5
#         out = FF.normalize(out,[-0.50008255/0.014866313,-0.4999794/0.014148686],[1/0.014866313,1/0.014148686])
#         out = torch.cat((out,elsePart2),2)
#         return out