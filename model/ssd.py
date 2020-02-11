import torch
improt torch.nn as nn
from vgg16 import *



class SSD(nn.Module):
    def __init__(self, class_num = 80):
        super(SSD, self).__init__()
        # Here asume the input size of image is 300*300*3
        self.vgg = vgg16(class_num = class_num)

    def forward(self, x):
        features = self.vgg(x)
        

    def predict(self, features, ):
        f1, f2, f3, f4, f5, f6 = features 
               
        
        
    def classifier(self, in_depth, class_num, anchor_num):
        """output is the feature n*m*(anchor_number*(class_num+4))
             anchor_num: the number of anchors in this feature.
             class_num: the number of class, the output is the possiblity of each class
                        (0~1) and ssd has one more class which is for background.
             4: the location variables (cx,cy,width, height) 
        """
        out_depth = anchor_num * (class_num + 4)
        layers = nn.Sequential(
            nn.Conv2d(in_depth, out_depth, 3, 1, 1),
            nn.ReLU(input=True)
        )
        return layers

