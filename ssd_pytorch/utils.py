import torch
import torch.nn as nn


class AnchorBox(Object):
    def __init__(self):
        #All these settings are based on ssd300 model
        self.image_size = 300
        self.feature_size = [38, 19, 10, 5, 3, 1] #the size of 6 output features
        self.anchor_num = [4, 6, 6, 6, 4, 4] #anchor number of each feature
        # anchor boxes min_size. In paper, there are fomulas to calculate themm.
        # the first size 30 for feature 38*38 is a specific one. Details in paper
        self.min_size = [30, 60, 111, 162, 213, 264] 
        self.max_size = [60, 111, 162, 213, 264, 315] # anchor boxes
        #the filter sizes. For instance, feature 38*38 is after filter 8 which is 300/8
        # Since the cx, cy, w, h are the rate of image_size, we use this to help calculate.
        self.step = [8, 16, 32, 64, 100, 300] #the filter size
        
    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_size):
            for i in range(f):
                for j in range(f):
                    step = self.step[k]
                    fk = self.image_size / self.step[k]
                     
                    # cx and cy are the rate of image. the real should be (i+0.5)*(image_size/feature_size)
                    # which is (i+0.5)*step[k]. Then the rate is (i+0.5)*step[k]/imgae_size which is (i+0.5)/fk
                    cx = (j + 0.5) / fk # be careful, j is related to x
                    cy = (i + 0.5) / fk
                    
                    #aspect_ratio = 1, the small square
                    s_k = self.min_size[k] / self.image_size        
                    anchors += [cx, cy, s_k, s_k]
                    
                    # the large square
                    s_k_prime = sqrt(s_k * (self.max_size[k] / self.image_size ))
                    anchors += [cx, cy, s_k_prime, s_k_prime]

                    # w:h = 1:2 or 2:1 trangles
                    s_k2_h = s_k / sqrt(2)
                    s_k2_w = s_k * sqrt(2)
                    anchors += [cx, cy, s_k2_h, s_k2_w]
                    anchors += [cx, cy, s_k2_w, s_k2_h]

                    # when there are 6 anchor boxes, we need w:h=1:3 and 3:1 trangles
                    if self.anchor_num == 6:
                        s_k3_h = s_k / sqrt(3)
                        s_k3_w = s_k * sqrt(3)
                        anchors += [cx, cy, s_k3_h, s_k3_w]
                        anchors += [cx, cy, s_k3_w, s_k3_h]
        
        #turn list in to tensor and resize it in (anchor_num, 4), here is (8732, 4)
        output = torch.Tensor(anchors)
        output = output.view(-1, 4)
        return output



