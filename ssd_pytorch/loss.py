import torch
import torch.nn as nn

__all__ = []

class MultiBoxLoss(nn.Module):
    def __init__(self, use_gpu, iou_threshold, variance,class_num):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.class_num = class_num
        self.iou_threshold = iou_threshold 
        self.bg_label = 0 #in class predictions, index=0 is the confidence for background
        self.variance = variance

    def forward(self, predictions, gt):
        #predictions = [loc, cls_conf, anchors]
        #loc:  output from SSD [batch_size, anchor_number, 4] ([batch_size, 8732, 4])
        #cls_conf: [batch_size,anchor_number, class_num] (batch_size, 8732, 21)
        #anchors: [8732, 4]
        #gt: ground truth, the location info of groud truth [batch_size, object_num, 5]
        #    "object_num" is the number of objects in a image. "5" mean 4 loaction info + 1 class

        loc, cls_conf, anchors = predictions
        batch_size = loc.size(0)
        anchors = anchors[:loc.size(1), :] #no changes, in case of multi-gpu
        anchor_num = anchors.size(0)

        loc_t = torch.Tensor(batch_size, anchor_num, 4)
        cls_t = torch.Tensor(batch_size, anchor_num)
        for  idx in range(batch_size):
            box_gt = gt[idx][:,:-1].data #[object_num, 4]
            cls_gt = gt[idx][:,-1].data  #[object_num]
            default_anchors = anchors.data
            # loc_t, cls_t will change inside "box_matcg"
            box_match(self.iou_threshold, self.variance, box_gt, cls_gt, default_anchors, loc_t, cls_t, idx) 


def box_match(thres, variance, box_gt, cls_gt, anchors, loc_t, cls_t, idx):
    """
    thres: iou_threshold between ground_truth boxes and anchors
    variance: the variance of encoding process
    box_gt: [object_num, 4] the location info of objects in this image
    cls_gt: [object_num] class No. of each object
    anchor: [anchors_num, 4] (8732, 4) default anchors' locations
    loc_t: [batch_size, anchors_num, 4](batch, 8732, 4) the output of match boxes location info
    cls_t: [batch_size, anchors_num](batc, 8732) the output of match boxes class
    idx: image index in batch_size of images"""
    # tranfer anchors from [cx,cy, w,h] to 4 points [xmin, ymin, xmax, ymax]
    anchors_4p = torch.cat((anchors[:, :2]-anchors[:,2:]/2, anchors[:,:2]+anchors[:,2:]/2) ,1)
    overlap = iou_jaccard(box_gt, anchors_4p)

def iou_jaccard(box_t, box_a):
    '''
    box_t: [object_num, 4]
    box_a: [anchor_num, 4] is [8732, 4]
    return [object_num, anchor_num] the iou of object_box to every anchor_box
    '''
    len_t = box_t.size(0)
    len_a = box_a.size(0)
    xy_min = torch.max(
           box_t[:,:2].unsqueeze(1).expand(len_t, len_a, 2),
           box_a[:,:2].unsqueeze(0).expand(len_t, len_a, 2)
    ) 
    xy_max = torch.max(
           box_t[:,:2].unsqueeze(1).expand(len_t, len_a, 2),
           box_a[:,:2].unsqueeze(0).expand(len_t, len_a, 2)
    ) 

def encode():




