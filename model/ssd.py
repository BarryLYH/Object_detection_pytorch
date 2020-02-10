import torch
improt torch.nn as nn


class SSD(nn.Module):
    def __init__(self, class_num = 80):
        super(SSD, self).__init__()
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
        self.prediction = self.

    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4_3(x)
        features.append(x)
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


    def predict(self, features, ):
        
        
        
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

