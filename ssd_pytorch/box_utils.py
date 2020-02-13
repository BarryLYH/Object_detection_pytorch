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


class Detector(Object):
    def __init__(self, class_num):
        self.threshold = 0.5
        self.top_k = 100
        self.class_num = class_num
        

    def forward(self, loc, cls_conf, anchors):
        # loc: the prediction of locations [batch_size, anchor_num, 4] which is [batch_size, 8732, 4]
        # cls_conf: the preidction of class confidence [batch_size, anchor_num, class_num] which is [batch_size, 8732, 81]
        # anchors: the default boxes [8732, 4]        
        batch_size = loc.size(0)
        #reorder cls_conf to [batch_size, class_num, anchor_num]
        cls_pred = cls_conf.permute(0,2,1)  
        for i in range(batch_size):
            decoded_box = decode(loc[i], anchors, self.variance)# [8732, 2]
            cls_scores = cls_conf[i].clone #class confidence of each box in image i, [81, 8732]

            for j in range(1, class_num):# j =0 is the background confidence
                # cls_score[j] is size [8732], cls_mask is also a tenor with False or Ture in size 8732
                # which mean the box of image_i, have class_i score over threshold 
                cls_mask = torch.gt(cls_scores[j], self.threshold)
                # get all the score which is larger than threshold, in order to to nms next
                scores = cls_scores[j][cls_mask] 
                
                if scores.dim() == 0: continue
                #the size of decoded_box is [8732, 4], cls_mask.unsqueeze(1), make is [8732, 1]. However, it is still 
                # not the same as decoded_box, so we need to .expand_as(decoded_box), loc_mask is [8732, 4]
                # which is, for exmaple, [[True, True, True, True], [False, False, False, False]....] 
                loc_mask = cls_mask.unsqueeze(1).expand_as(decoded_box)
                boxes = decoded_box[loc_mask].view(-1, 4) 
                
                #left = torchvision.ops.nms(boxes, scores, nms_threshold)
                b, s = nms(boxes, scores, self.threshold, self.top_k)

            
                                  
def decode(loc, anchors, variance):
    #loc: [8732, 4] , the 4 here is the bias rate of anchor box.
    #     so that we use "decode" to transfer bias to predicted box
    #anchors: [8732, 4]
    #variance: in paper, this is the manual set variance to decode
    boxes = torch.cat(
            (anchors[:, :2] + anchors[:, 2:] * loc[:, :2]* variance[0],
             anchors[:, 2:] * torch.exp(loc[:, 2:] * variance[1]))  ,1)
    # transfer box from (cx, cy w, h) to (xmin, ymin, xmax, ymax)
    boxes[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes

def nms(boxes, scores, threshold, top_k):
    # implement non-maxmium suppression to avoid too many overlapping
    # bounding box
    # boxes: [boxes_num, 4]
    # scores: [boxes_num]
    # threshold: IOU threshold 
    keep = []
    if scores.size(0) == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area  = (x2 - x1) * (y2 - y1)
    _, order = scroes.sort(0)
    while order.size(0) > 0:
        if order.size(0) == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)
        xx1 = x1[order[1:]].clamp(min = x1[i])
        yy1 = y1[order[1:]].clamp(min = y1[i])
        xx2 = x2[order[1:]].clamp(max = x2[i])
        yy2 = y2[order[1:]].clamp(max = y2[i])
        overlap = (xx2-xx1).clamp(min=0) * (yy2- yy1).clamp(min=0)
        iou = overlap / (area[i] + area[order[1:]]-overlap)
        iou_mask = iou.le(threshold)
        order = order[1:][iou_mask] # [len(order-1)]

    if len(keep) > top_k:
        keep = keep[:top_k]
    output_boxes = boxes[keep[:]]
    output_scores = scores[keep[:]]
    return output_boxes, output_scores
