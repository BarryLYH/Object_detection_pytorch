import torch
import torch.nn as nn
import torch.nn.functional as F


class SSD(nn.Module):
    def __init__(self, phase = "train",class_num = 80):
        super(SSD, self).__init__()
        # Here asume the input size of image is 300*300*3
        self.class_num = class_num
        self.backbone = VGG16(class_num = self.class_num)#output of this layer are 6 features
        self.predict = Predict(phase=self.phase, class_num = self.class_num)

    def forward(self, x):
        features = self.backbone(x)
        loc, cls = self.predict(features) 
        # loc and cls shapes(batch_size, all_anchor_boxes_num, 4),(batch_size, all_anchor_boxes_num, class_num+1)  
        # the final output is the predictions of each feature cells.
        # loc is the location predictions for each anchor box and 
        # cls is the class predictions for eahc anchor boxs
        return loc, cls


class Predict(nn.Module):
    def __init__(self, phase, class_num = 80):
        self.in_planes = [512, 1024, 512, 256, 256, 256]
        self.anchors = [4, 6, 6, 6, 4, 4]
        self.class_num = class_num + 1
        self.phase = phase

    def forward(self, features):
        loc = []
        cls = []
        # the default shape of pytorch is (N, C, H ,W) and in this case it is 
        # (batch_size, box_loaction, feature_height, feature_width). For instance,
        # if is feature 38*38*(4*4), the ourput will be (batch_size, 4*4, 38, 38) 
        for i,f in enumerate(features):
            loc.append(nn.Conv2d(self.in_planes[i], self.anchors[i]*4, 3, 1, 1))
            cls.append(nn.Conv2d(self.in_planes[i], self.anchors[i]*self.class_num+1, 3, 1, 1))

        location = []
        clsconf = []
        # This step will change (N, C, H, W) to (N, H, W, C) so that we can easily concatenate all
        # boxes or classes
        # using contiguous to make sure all tensors are continuous in RAM and we can us view next
        for l, c in zip(loc, cls):
             location.append(l.permute(0,2,3,1).contiguous())
             clsconf.append(c.permute(0,2,3,1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in location], 1)
        cls = torch.cat([o.view(o.size(0), -1) for o in clsconf], 1)
        #After above step, all the predictions will be concatenated. Before concatenating, the shapes
        # are loc(batch, H, W, anchor_num*4) and cls(batch, H, W, anchor_num*class_num). Now they are
        #loc(batch, all_prebox_num*4) and cls(batch, all_prebox_num*class_num) 

        # Here, we resize loc and cls to (batch_size, boxes_num, 4) and (batch_size, boxes_num, class_num)
        # which are (batch_size, 8732, 4) and (batch_size, 8732, 81)
        loc = loc.view(loc.size(0), -1, 4)
        cls = cls.view(cls.size(0), -1, self.class_num)
        return loc, cls

class VGG16(nn.Module):
    def __init__(self, class_num = 80):
        super(VGG16, self).__init__()
        # Here asume the input size of image is 300*300*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), #300*300*64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), #300*300*64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) #150*150*64, maxpool1
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), #150*150*128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), #150*150*128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) #75*75*64, maxpool2
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), #75*75*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), #75*75*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), #75*75*256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) #38*38*256, maxpool3
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), #38*38*512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), #38*38*512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), #38*38*512 Here is feature1
            nn.ReLU(inplace=True),
        )
        self.maxpool4_3 = nn.MaxPool2d(2,2)  #19*19*512
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), #19*19*512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), #19*19*512
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), #19*19*512
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1) #19*19*512
        ) 
        # vgg16 model ends here. There is no fc, instead conv6 and conv7
        #take the place of fx    
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1), #19*19*1024
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(1024, 1024, 1, 1), #19*19*1024, Here is feature2
            nn.ReLU(inplace=True)
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, 1), #19*19*256
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1), #10*10*512, here is feature3
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 128, 1, 1), #10*10*128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), #5*5*256, here is feature4
            nn.ReLU(inplace=True)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(256,128, 1, 1), #5*5*128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1), #3*3*256, here is feature5
            nn.ReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(256,128, 1, 1), #3*3*128
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1), #1*1*256, here is feature6
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4_3(x)
        features.append(F.normalizex(x, p=2, dim=1, eps=1e-12)) #L2normalization in torch
        x = self.maxpool4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        features.append(x)
        x = self.conv8(x)
        features.append(x)
        x = self.conv9(x)
        features.append(x)
        x = self.conv10(x)
        features.append(x)
        x = self.conv11(x)
        features.append(x)
        return features
        
def ssd(**kwarg):
    model = SSD(**kwarg)
    return model

    

