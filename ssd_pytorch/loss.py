import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#__all__ = []

class MultiBoxLoss(nn.Module):
    def __init__(self, use_gpu, iou_threshold, variance,class_num, negpos_ratio):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.class_num = class_num
        self.iou_threshold = iou_threshold 
        self.bg_label = 0 #in class predictions, index=0 is the confidence for background
        self.variance = variance
        self.negpos_ratio = negpos_ratio # the ratio of negative_sample/positive_sample

    def forward(self, predictions, gt):
        #predictions = [loc, cls_conf, anchors]
        #loc:  output from SSD [batch_size, anchor_number, 4] ([batch_size, 8732, 4])
        #cls_conf: [batch_size, anchor_number, class_num] (batch_size, 8732, 21)
        #anchors: [8732, 4]
        #gt: ground truth, the location info of groud truth [batch_size, object_num, 5]
        #    "object_num" is the number of objects in a image. "5" mean 4 loaction info 
        # [xmin, ymin, xmax, ymax] + 1 class

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
            # loc_t, cls_t will change inside "box_match()"
            box_match(self.iou_threshold, self.variance, box_gt, cls_gt, default_anchors, loc_t, cls_t, idx) 
        if self.use_gpu:
            loc_t.cuda()
            cls_t.cuda()
        loc_t = Variable(loc_t, requires_grad=False)
        cls_t = Variable(cls_t, requires_grad=False)
        #find anchors whose classes are not background
        #cls_pos: the positive samples [batch_size, anchor_num]
        cls_pos = (cls_t > 0) 

        # location loss function, use smoothL1 loss function
        # loc_t, loc: [batch_size, anchor_num ,4]
        # cls_pos:   [batch_size, anchor_num]
        box_mask = cls_pos.unsqueeze(1).expand(batch_size, anchor_num, 4)
        loc_p = loc[box_mask].view(-1, 4) 
        loc_gt = loc_t[box_mask].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_gt, reduction='mean')

        # class confidence loss function, crossentrypy
        # cls_conf: [batch, num_priors, num_classes]
        # cls_t: [batch, num_priors]
        cls_conf = cls_conf.view(-1, self.class_num) # (batch*8732, 21)
        # Do log(softmax([num_classes]))
        cls_log = -torch.log(F.softmax(1, cls_conf)) 
        # the prediction loss of anchor box. We get the prediction log(softmax)
        # of the expected class. This class's probability should be 1, after log(1)
        # it should be 0. so the distance is 0 - log(softmax[])
        cls_b_loss = torch.gather(cls_log, 1, cls_t.view(-1, 1))
        #mine the negative samples
        cls_b_loss[cls_pos.view(-1, 1)] = 0 # we turn all the positve samples' losses to 0
        cls_b_loss = cls_b_loss.view(batch_size, -1) # [batch*anchor, 1] -> [batch, anchors]
        #after these two sort, neg_index will be the index of each element in descending order
        #for example, [2,3,4,1] the descending order should be [4,3,2,1]and the neg_index
        # will be [2,1,0,4] (4 is largest, so in decsending order the index is 0)
        _, loss_index = cls_b_loss.sort(1, descending=True) #[batch_size, anchor_num]
        _, neg_index = loss_index.sort(1) #[batch_size, anchor_num]
        pos_num = cls_pos.long().sum(1, keepdim=True) #[batch_size, 1]
        neg_num = torch.clamp(self.negpos_ratio*pos_num, max=cls_pos.size(1)-1)#[batch_size, 1]
        cls_neg = neg_index < neg_num.expand_as(neg_index)  #[batch_size,anchor_num]

        cls_pos_expand = cls_pos.unsquence(2).expand_as(cls_conf) #(batch, anchor_num, class_num) Ture or False inside
        cls_neg_expand = cls_neg.unsquence(2).expand_as(cls_conf) #(batch, anchor_num, class_num) Ture or False inside
        cls_pre = cls_conf[(cls_pos_expand+cls_neg_expand)>0] #[batch, anchor_num, class_num]
        cls_pre = cls_pre.view(-1, self.class_num)#[batch*anchor, class_num]
        cls_target = cls_t[(cls_pos+cls_neg)>0].view(-1) # (batch*anchor)
        loss_c = F.cross_entrypy(cls_pre, cls_target, size_average=False)
        N = pos_num.data.sum()
        loss_c = loss_c / N
        loss_l = loss_l / N
        return loss_l, loss_c

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
    area_t = (box_t[:,2] - box_t[:,0]) * (box_t[:,3] - box_t[:,1]).unsqueeze(1).expand(overlap)
    area_a = (box_a[:,2] - box_a[:,0]) * (box_a[:,3] - box_a[:,1]).unsqueeze(0).expand(overlap)
    iou = overlap / (area_t + area_a - overlap)
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




