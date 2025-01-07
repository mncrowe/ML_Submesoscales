# selection of pytorch CNN models

import numpy as np
import torch
import math
import torch.nn as nn
#from torchsummary import summary

import torch.nn.functional as F

from typing import Optional, Tuple, Union, List
#from labml_helpers.module import Module

##########################
##### basic CNN ##########
##########################

class Autoencoder(nn.Module):
  def __init__(self,Nl_in=1,Nl_out=1,N_out=32,N_max=2,N_conv=3,Activ=nn.ReLU()):
    super().__init__()
    
    # Activation functions   
    self.activ = Activ
    self.sigmoid = nn.Sigmoid()

    # Encoder layers
    self.refpad1 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv1 = nn.Conv2d(Nl_in, N_out, N_conv,padding=0)
    self.maxpool1 = nn.MaxPool2d(N_max, padding=0)
    self.refpad2 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv2 = nn.Conv2d(N_out, N_out, N_conv, padding=0)
    self.maxpool2 = nn.MaxPool2d(N_max, padding=0)
    
    # Decoder layers
    self.refpad3 = nn.ReflectionPad2d(np.int32(((N_conv-1)/2,0,(N_conv-1)/2,0)))
    self.deconv1 = nn.ConvTranspose2d(N_out, N_out, N_conv, stride=N_max, padding=(N_conv-1), output_padding=1)
    self.refpad4 = nn.ReflectionPad2d(np.int32(((N_conv-1)/2,0,(N_conv-1)/2,0)))
    self.deconv2 = nn.ConvTranspose2d(N_out, N_out, N_conv, stride=N_max, padding=(N_conv-1), output_padding=1)
    self.conv_out = nn.ConvTranspose2d(N_out, Nl_out, N_conv, padding=1)

  def forward(self, x):
    x = self.maxpool1(self.activ(self.conv1(self.refpad1(x))))
    x = self.maxpool2(self.activ(self.conv2(self.refpad2(x))))
    x = self.deconv1(self.refpad3(self.activ(x)))
    x = self.deconv2(self.refpad4(self.activ(x)))
    x = self.sigmoid(self.conv_out(x))
    return x

###########################
### Sharpening CNN ########
###########################

class Sharpener(nn.Module):
  def __init__(self,Nl_in=1,Nl_out=1,N_out=256,N_max=2,N_conv=3,Activ=nn.ReLU()):
    super().__init__()
    
    # Activation functions   
    self.activ = Activ    
    
    # Encoder layers
    self.refpad1 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv1 = nn.Conv2d(Nl_in, N_out, N_conv,padding=0)
    self.refpad2 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv2 = nn.Conv2d(N_out, N_out//2, N_conv, padding=0)
    self.maxpool = nn.MaxPool2d(N_max, padding=0)
    self.refpad3 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv3 = nn.Conv2d(N_out//2, N_out//4, N_conv, padding=0)    
    
    # Decoder layers
    self.refpad4 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv4 = nn.Conv2d(N_out//4, N_out//4, N_conv, padding=0)    
    self.upsample = nn.Upsample(scale_factor=N_max, mode='nearest')
    self.refpad5 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv5 = nn.Conv2d(N_out//4, N_out//2, N_conv, padding=0)
    self.refpad6 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv6 = nn.Conv2d(N_out//2, N_out, N_conv, padding=0)
    self.refpad_out = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv_out = nn.Conv2d(N_out, Nl_out, N_conv, padding=0)
  
  def forward(self, x):
    x = self.activ(self.conv1(self.refpad1(x)))
    x = self.activ(self.conv2(self.refpad2(x)))
    x = self.maxpool(x)
    x = self.activ(self.conv3(self.refpad3(x)))
    x = self.activ(self.conv4(self.refpad4(x)))
    x = self.upsample(x)
    x = self.activ(self.conv5(self.refpad5(x)))
    x = self.activ(self.conv6(self.refpad6(x)))
    x = self.conv_out(self.refpad_out(x))
    return x
    
###########################
### Bolton & Zanna ########
###########################
    
class BoltonZanna(nn.Module):
  def __init__(self,N_out=256,N_max=2,N_conv=3,Activ=nn.ReLU()):
    super().__init__()
    
    
    
  def forward(Self, x):
    
    return x

###############################
#### Classification CNN #######
###############################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
#################################
##### UNET (of doom) ############
#################################

# works for various data sizes, e.g. powers of 2 > 16

class conv_block(nn.Module):
  def __init__(self, N_in=1, N_out=32, N_conv=3, Activ=nn.ReLU()):
    super().__init__()
    self.refpad1 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv1 = nn.Conv2d(N_in, N_out, N_conv, padding=0)
    self.bn1 = nn.BatchNorm2d(N_out)
    self.refpad2 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.conv2 = nn.Conv2d(N_out, N_out, N_conv, padding=0)
    self.bn2 = nn.BatchNorm2d(N_out)
    self.activ = Activ

  def forward(self, inputs):
    x = self.refpad1(inputs)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activ(x)
    x = self.refpad2(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.activ(x)
    return x

class downsample_block(nn.Module):
  def __init__(self, N_in=1, N_out=32, N_max=2, p_skip=0.0, N_conv=3, Activ=nn.ReLU()):
    super().__init__()
    self.dblconv1 = conv_block(N_in,N_out,N_conv,Activ)
    self.maxpool1 = nn.MaxPool2d(N_max)
    self.dropout1 = nn.Dropout(p_skip)

  def forward(self,inputs):
    x = self.dblconv1(inputs)
    y = self.maxpool1(x)
    y = self.dropout1(y)
    return x, y

class upsample_block(nn.Module):
  def __init__(self, N_in=1, N_out=32, N_max=2, p_skip=0.0, N_conv=3, Activ=nn.ReLU()):
    super().__init__()
    #self.refpad1 = nn.ReflectionPad2d(np.int32(((N_conv-1)/2,0,(N_conv-1)/2,0)))
    self.refpad1 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.invconv1 = nn.ConvTranspose2d(N_in, N_out, N_conv, stride=N_max, padding=N_conv, output_padding=1)
    self.dblconv1 = conv_block(2*N_out,N_out,N_conv,Activ)
    
  def forward(self,inputs,skip):
    x = self.refpad1(inputs)
    x = self.invconv1(x)
    x = torch.cat([x, skip], axis=1)
    x = self.dblconv1(x)
    return x

class UNET(nn.Module):
  def __init__(self,N_in=1,N_out=1,N_up=(32,64,128,256),N_down=(512,256,128,64,32),N_conv=3,N_max=2,Activ=nn.ReLU(),Activ_out=nn.Sigmoid()):
    super().__init__()
    self.encoder1 = downsample_block(N_in,N_up[0])
    self.encoder2 = downsample_block(N_up[0],N_up[1])
    self.encoder3 = downsample_block(N_up[1],N_up[2])
    self.encoder4 = downsample_block(N_up[2],N_up[3])
    self.bottleneck = conv_block(N_up[3],N_down[0])
    self.decoder1 = upsample_block(N_down[0],N_down[1])
    self.decoder2 = upsample_block(N_down[1],N_down[2])
    self.decoder3 = upsample_block(N_down[2],N_down[3])
    self.decoder4 = upsample_block(N_down[3],N_down[4])
    self.refpad1 = nn.ReflectionPad2d(np.int32((N_conv-1)/2))
    self.outconv = nn.Conv2d(N_down[4], N_out, N_conv, padding=0)
    self.sigmoid = Activ_out

  def forward(self,inputs):
    x1, y = self.encoder1(inputs)
    x2, y = self.encoder2(y)
    x3, y = self.encoder3(y)
    x4, y = self.encoder4(y)
    y = self.bottleneck(y)
    y = self.decoder1(y,x4)
    y = self.decoder2(y,x3)
    y = self.decoder3(y,x2)
    y = self.decoder4(y,x1)
    y = self.refpad1(y)
    y = self.outconv(y)
    y = self.sigmoid(y)
    return y
    
#################################
####### Example UNET ############
#################################

# stolen Unet only works for certain sized data, e.g. (1, 284, 284)

class UNET_2(nn.Module):
  def contracting_block(self, in_channels, out_channels, kernel_size=3):
    block = torch.nn.Sequential(
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(out_channels),
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(out_channels),
      )
    return block
    
  def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
    block = torch.nn.Sequential(
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),
      torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
      )
    return  block
    
  def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
    block = torch.nn.Sequential(
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(mid_channel),
      torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(out_channels),
      )
    return  block

  def __init__(self, in_channel, out_channel):
    super(UNET_2, self).__init__()
    #Encode
    self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
    self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv_encode2 = self.contracting_block(64, 128)
    self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
    self.conv_encode3 = self.contracting_block(128, 256)
    self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
    # Bottleneck
    self.bottleneck = torch.nn.Sequential(
      torch.nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(512),
      torch.nn.Conv2d(kernel_size=3, in_channels=512, out_channels=512),
      torch.nn.ReLU(),
      torch.nn.BatchNorm2d(512),
      torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
      )
      # Decode
    self.conv_decode3 = self.expansive_block(512, 256, 128)
    self.conv_decode2 = self.expansive_block(256, 128, 64)
    self.final_layer = self.final_block(128, 64, out_channel)
        
  def crop_and_concat(self, upsampled, bypass, crop=False):
    if crop:
      c = (bypass.size()[2] - upsampled.size()[2]) // 2
      bypass = F.pad(bypass, (-c, -c, -c, -c))
      return torch.cat((upsampled, bypass), 1)
    
  def forward(self, x):
    # Encode
    encode_block1 = self.conv_encode1(x)
    encode_pool1 = self.conv_maxpool1(encode_block1)
    encode_block2 = self.conv_encode2(encode_pool1)
    encode_pool2 = self.conv_maxpool2(encode_block2)
    encode_block3 = self.conv_encode3(encode_pool2)
    encode_pool3 = self.conv_maxpool3(encode_block3)
    # Bottleneck
    bottleneck1 = self.bottleneck(encode_pool3)
    # Decode
    decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
    cat_layer2 = self.conv_decode3(decode_block3)
    decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=True)
    cat_layer1 = self.conv_decode2(decode_block2)
    decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
    final_layer = self.final_layer(decode_block1)
    return  final_layer


##################################
##### Testing and Summary ########
##################################

# Create an instance of the model
#model = Autoencoder()
#model.cuda() # send the model to the GPU (*waves goodbye*)

#Print the model summary
#summary(model, (1, 32, 32))
