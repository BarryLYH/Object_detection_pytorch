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
    
    #get the large iou area value and the index of anchors
    #best_an_area: [object_num]
    #best_an_index: [object_num] the index of anchors whose overlap area of the object is largest
    best_an_iou, best_an_index = overlap.max(1, keepdim = False) 
    #best_gt_area: [anchor_num] -> [8732]
    #best_gt_index: [anchor_num] -> [8732]
    best_gt_iou, best_gt_index = overlap.max(0, keepdim = False)

    #best_an_index: [object_num]
    #best_gt_iou: [anchor_num] -> [8732]
    # for_loop here, we make sure to object_j's ground truth, the iou among all anchor
    # matches the anchor's best choice. For instance, obj_j should match anchor_i in best_an_index.
    # However, anchor_i matches obj_k in best_gt_index. In this case, we force obj_j matches
    #anchor_i and make their overlap to be 2.0. Since anchors are much more than ground_truths,
    # we ignore the case obj_j and obj_k match anchor_i at the same time.
    best_gt_iou.index_fill_(0, best_an_index, 2) # this line is the same of the commented code in next loop
    for j in range(best_an_index.size(0)): # range:0~object_nun-1
        best_gt_index[best_an_index[j]] = j
        #best_gt_iou[best_an_index[j]] = 2.0

    #best_gt_index is [8732] which is the index of objects inside meaning the matched gt_index 
    # of each anchor. 
    #box_gt: [object_num, 4] the ground truth of objects' boxes
    box_matches = box_gt[best_gt_index] #[anchor_num, 4] since anchor_num >> object_num, lots of repeating boxes inside
    cls_matches = cls_gt[best_gt_index]+1 #[anchor_num], since cls_gt is from 0~obj_cls-1, and 0 in sdd is bg
    cls_matches[best_gt_iou < thres] = 0 # iou<thres, the cls should be backgound
    box_matches_encoded = encode(box_matches, anchors, variance) # encode box with anchor
    loc_t[idx] = box_matches_encoded # the matched and encoded box gt of image_idx
    cls_t[idx] = cls_matches # match cls gt of image_idx

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
    xy_max = torch.min(
           box_t[:,2:].unsqueeze(1).expand(len_t, len_a, 2),
           box_a[:,2:].unsqueeze(0).expand(len_t, len_a, 2)
    )
    #get the width and height of the overlap areas
    box_wh = torch.clamp((xy_max-xy_min), min = 0) # [object_num, anchor_num, 2]
    overlap = torch.zeros(len_t, len_a)
    # calculate iou overlap area
    overlap[:,:] = box_wh[:,:,0] * box_wh[:,:,1] #[object_num, anchor_num]

    #TODO iou = 

    return iou

def encode(bow_match, anchors, variance):
    #bow_match: [anchor_num, 4] and "4" is xmin,ymin,xmax,ymax, which is ground_truth position
    #anchors: [anchor_num, 4]
    #variance: [2]
    g_xy = ((bow_match[:, :2] + bow_match[:, 2:]) / 2 - anchors[:, 2:]) / (anchors[:, 2:] * variance[0])
    g_wh = torch.log((bow_match[:, 2:] - bow_match[:, :2])  / anchors[:, 2:])/variance[1]
    #g_xy: [anchor_num, 2]
    #g_wh: [anchor_num, 2]
    return torch.cat((g_xy, g_wh),1)




