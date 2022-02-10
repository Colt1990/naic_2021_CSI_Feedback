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
    






class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        latent_dim = 128
        self.feedback_bits = feedback_bits

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

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(4, 2,
                              kernel_size= 3, stride= 1, padding  = 1),

        )
        
        self.fc_mu = nn.Linear(15360, int(feedback_bits / self.B))  #32256  #

        #self.norm = nn.BatchNorm1d(2048)
        self.sa = SpatialGate()
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.hsig= hsigmoid()
        self.drop = nn.Dropout(0.1)

        
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 

    def forward(self, x):
        
        x =x[:,:,:60,:]
        x = FF.normalize(x,[0.50008255,0.4999794],[0.014866313,0.014148686])

        result1 = self.encoder1(x)
        result2 = self.encoder2(x)
        result = torch.cat((result1,result2),dim=1)
        result =self.encoder_conv(result)
        #print(result.shape)
        
        result = torch.flatten(result, start_dim=1)
        
        
        mu = self.fc_mu(result)
        out = self.sig(mu)

        if self.quantization:
            out = self.quantize(out)
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
  
        
        self.decoder_input = nn.Linear(int(self.feedback_bits / self.B), 15360)   #
    
        
        self.refine = RefineBlock()
        hidden_dims =[32,128,256,512,1024,512,256,128,2]
        hidden_dims.reverse()
        modules = []

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

        else:
            out = x
        out = out.view(-1, int(self.feedback_bits / self.B)) #int(self.feedback_bits / self.B)
        #print(out.shape)
        #out = out.half()
        result = self.decoder_input(out)
        result = result.view(-1,2, 60, 128)
        
     
        result1 = self.decoder1(result)
        result2 = self.decoder2(result)

        out = torch.cat([result1,result2],1)

        out = self.final_layer(out)
        
        elsePart2 = torch.ones([out.shape[0],2,66,128],device=out.device)*0.5

        out = FF.normalize(out,[-0.50008255/0.014866313,-0.4999794/0.014148686],[1/0.014866313,1/0.014148686])

        
        out = torch.cat((out,elsePart2),2)

        return out


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


