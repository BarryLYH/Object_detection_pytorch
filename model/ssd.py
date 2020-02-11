import torch
improt torch.nn as nn
from vgg16 import *



class SSD(nn.Module):
    def __init__(self, cfg, class_num = 80):
        super(SSD, self).__init__()
        # Here asume the input size of image is 300*300*3
        self.backbone = vgg16(class_num = class_num)
        self.predict = predict(cfg)        


    def forward(self, x):
        features = self.backbone(x)
        


class Predict(nn.Module):
    def __init__(self, class_num = 80):
        self.in_planes = [512, 1024, 512, 256, 256, 256]
        self.anchors = [4, 6, 6, 6, 4, 4]
        self.class_num = class_num + 1

    def forward(self, features):
        los = []
        cls = []
        for i,f in enumerate(features):
            los.append(nn.Conv2d(self.in_planes[i], self.anchors[i]*4, 3, 1, 1))
            cls.append(nn.Conv2d(self.in_planes[i], self.anchors[i]*self.class_num+1, 3, 1, 1))
        
        return         

        
        
        

cfg = [[4, 1024], [],6,6,]

def ssd(**kwarg):
    model = SSD(**kwarg)
    return model

def predict():
    model = Predict()
    return model
    

