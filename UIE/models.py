import argparse
import os

import numpy as np

#import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


# Channel and Spatial Attention
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
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

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k,s,p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))



class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.PixelShuffle(2)#nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c,nf=64,nb=23,kernel_size=1,padding=0)
        self.conv1 = conv_block(in_c, in_c*4,nf=64,nb=23,kernel_size=1,padding=0)

    def forward(self, inputs, skip):
        #print(inputs.shape)
        x = self.conv1(inputs)
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x   



class conv_block(nn.Module):
    def __init__(self, in_nc, out_nc,  nf, nb,kernel_size,padding, gc=32, upscale=4, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
            
        super().__init__()

        self.conv1 = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
        self.bn1 = nn.BatchNorm2d(out_nc)

        self.relu = nn.PReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        return x
        
class encoder_block(nn.Module):
    def __init__(self, in_nc, out_nc):
        super().__init__()

        self.conv1 =  conv_block(in_nc, out_nc,nf=64,nb=23,kernel_size=1,padding=0)
        self.conv2 =  conv_block(in_nc, out_nc,nf=64,nb=23,kernel_size=1,padding=0)
        self.conv3 =  conv_block(in_nc, out_nc,nf=64,nb=23,kernel_size=1,padding=0)
        self.conv4 =  conv_block(in_nc, out_nc,nf=64,nb=23,kernel_size=1,padding=0)

        self.conv = conv_block(in_nc*8, out_nc*4,nf=64,nb=23,kernel_size=1,padding=0)
        
        self.pool = nn.PixelUnshuffle(2)#nn.MaxPool2d((2, 2))
                

    def forward(self, x):
       
        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        branch4 = self.conv4(x)  
        
        op1 = torch.cat([branch1,branch2],1) 
        op2 = torch.cat([op1,branch3],1)
        op3 = torch.cat([op2,branch4],1)     

        p = self.pool(op3)
        p = self.conv(p)
        
        return op3,p


class LitNet(nn.Module):

    def __init__(self):
        super(LitNet, self).__init__()

        scale=1
        self.layer1_1 = Conv2D_pxp(1, 32, 3,1,1)
        self.layer1_2 = Conv2D_pxp(1, 32, 5,1,2)
        self.layer1_3 = Conv2D_pxp(1, 32, 7,1,3)

        self.layer_rgb = Conv2D_pxp(3, 16, 1,1,0)

        self.local_attn_r = CBAM(48)
        self.local_attn_g = CBAM(48)
        self.local_attn_b = CBAM(48)

        self.local_attn_s3 = SpatialGate()
        self.local_attn_s2 = SpatialGate()
        self.local_attn_s1 = SpatialGate()
        self.local_attn_l = CBAM(208)

        self.layer3_1 = Conv2D_pxp(144, 64, 3,1,1)

        #Encoder 
        self.e1 = encoder_block(64, 32)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.e2 = encoder_block(64, 32)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.e3 = encoder_block(64, 32)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
    

        #Bottleneck 
        self.b = conv_block(64, 64,nf=64,nb=23,kernel_size=1,padding=0)

        #Decoder
        self.d1 = decoder_block(64, 64)
        self.d2 = decoder_block(64,64)
        self.d3 = decoder_block(64,64)

        #self.up = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=2, stride=2, padding=0)

        self.convf = Conv2D_pxp(208, 64, 3,1,1)
        self.convf_1 = Conv2D_pxp(64, 3, 3,1,1)

    def forward(self, input):
        input_1 = torch.unsqueeze(input[:,0,:,:], dim=1)
        input_2 = torch.unsqueeze(input[:,1,:,:], dim=1)
        input_3 = torch.unsqueeze(input[:,2,:,:], dim=1)
        
        k = input

        l1_1=self.layer1_1(input_1) #n*32*400*400
        l1_2=self.layer1_2(input_2) #n*32*400*400       
        l1_3=self.layer1_3(input_3) #n*32*400*400

        input = self.layer_rgb(input)

        l1_1=self.local_attn_r(torch.cat((l1_1,input),1))

        l1_2=self.local_attn_g(torch.cat((l1_2,input),1))

        l1_3=self.local_attn_b(torch.cat((l1_3,input),1))

        #Input to layer 2- n*96*400*400
        input_l2=torch.cat((l1_1,l1_2),1)
        
        input_l2=torch.cat((input_l2,l1_3),1)

        later_add = input_l2

        input_l2 = self.layer3_1(input_l2)

        #Encoder
        s1,p1 = self.e1(input_l2)
        
        s1 = self.conv1(s1)
        p1 = self.conv1(p1)

        
        s2,p2 = self.e2(p1)
         
       
        s2 = self.conv2(s2)
        p2 = self.conv2(p2)

        s3,p3 = self.e3(p2)
         
       
        s3 = self.conv3(s3)
        p3 = self.conv3(p3)

        
        b = self.b(p3)
        
        s3=self.local_attn_s3(s3)
        d1 = self.d1(b, s3)
        s2=self.local_attn_s2(s2)
        d2 = self.d2(d1, s2)
        s1=self.local_attn_s1(s1)

        d3 = self.d3(d2, s1)

        final = self.local_attn_l(torch.cat((later_add,d3),1))

        final_output = self.convf(final)
        final_output = self.convf_1(final_output)
        final_output = final_output + k
  
        return final_output
