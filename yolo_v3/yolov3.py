import torch
import torch.nn as nn
import torch.nn.functional as F


class Convlayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernal_size, stride, padding):
        self.conv = nn.Conv2d(in_channel, out_channel, kernal_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ResOperator(nn.Module):
    def __init__(self, block_num, in_channel):
        super(ResOperator, self).__init__()
        self.block_num = block_num
        mid_channel = in_channel // 2
        self.conv1 = Convlayer(in_channel, 2*in_channel, 3, 2, 1)
        self.conv2 = Convlayer(2*in_channel, in_channel, 1, 1, 0)
        self.conv3 = Convlayer(in_channel, 2*in_channel, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.block_num):
            residual = x
            x = self.conv2(x)
            x = self.conv3(x)
            x = x + residual
           
        return x

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        # the input size of image is 256*256*3
        self.conv1 = Convlayer(3, 32, 3, 1, 1) #256*256*32
        self.resblock1 = ResOperator(1, 32, True) #128*128*64
        self.resblock2 = ResOperator(2, 64, True) #64*64*128
        self.resblock3 = ResOperator(8, 128, True) #32*32*256
        self.resblock4 = ResOperator(8, 256, True) #16*16*512
        self.resblock5 = ResOperator(4, 512, False) #8*8*1024
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        out3 = self.resblock3(x)
        out2 = self.resblock4(out3)
        out1 = self.resblock5(out2)
        return out1, out2, out3

class YoloBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(YoloBlock, self).__init__()
        self.layer1 = Convlayer(in_channel, out_channel, 1, 1, 0)
        self.layer2 = Convlayer(out_channel, 2 * out_channel, 3, 1, 1)
        self.layer3 = Convlayer(2 * out_channel, out_channel, 1, 1, 0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Yolov3(nn.Module):
    def __init__(self,):
        super(Yolov3, self).__init__()
        self.darknet53 = Darknet53()
        self.yoloblock512 = YoloBlock(1024, 512)
        self.yoloblock256 = YoloBlock(768, 256)
        self.yoloblock128 = YoloBlock(384, 128)

    def forward(self, x):
        # when training in torch, the dimension is (batch, c, h, w)
        # x: [batch_size, 3, 256, 256]
        #out1: [batch_size, 1024, 8, 8]
        #out2: [batch_size, 512, 16, 16]
        #out3: [batch_size, 256, 32, 32]
        out1, out2, out3 = self.darknet53(x)
        
        out1 = self.yoloblock(out1) # 8*8*512
        feature1 = Convlayer(512, 1024, 3, 1, 1)(out1) #8*8*1024
        feature1 = Convlayer(1024, 255, 1, 1, 0)(feature1) #8*8*255

        temp = Convlayer(512, 256, 1, 1, 0) #8*8*256
        temp = F.interpolate(input = temp, 
                             scale_factor = 2,
                             mode = "nearest") #16*16*256
        temp = torch.cat((temp, out2), dim=1) #16*16*768
        temp = self.yoloblock256(temp) #16*16*256
        feature2 = Convlayer(256, 512, 3, 1, 1)(temp) # 16*16*512
        feature2 = Convlayer(512, 255, 1, 1 ,1)(feature2) #16*16*255

        temp = Convlayer(256, 128, 1, 1, 0) #16*16*128
        temp = F.interpolate(input = temp, 
                             scale_factor = 2,
                             mode = "nearest") #32*32*128
        feature3 = torch.cat((temp, out3), dim=1) #32*32*384
        feature3 = self.yoloblock128(feature3) #32*32*128
        feature3 = Convlayer(128, 256, 3, 1, 1) #32*32*256
        feature3 = Convlayer(256, 255, 1, 1, 0) #32*32*255

        return feature1, feature2, feature3


        

        

