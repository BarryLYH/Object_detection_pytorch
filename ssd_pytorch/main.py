import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

from ssd import *
from data import * 
from loss import *
from box_utils import *

def adjust_learning_rate(optimizer, decay, step):
    lr = lr * (decay ** (step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def main():
    parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
    parser.add_argument("--epoch", default=10,
                        help="The epoch value")
    parser.add_argument("--data", default="/Users/Barry/Desktop/object_detection_pytorch/dataset.txt",
                        help="The file path of dataset")
    parser.add_argument("--class_num", default=80,
                        help="The class number of the dataset")
    parser.add_argument("--lr", default=0.01,
                        help="The default learning rate")
    args = parser.parse_args()
    epoch = args.epoch
    data_path = args.data
    cls_num = args.class_num
    use_gpu = torch.cuda.is_available()
    
    model = ssd(class_num = cls_num + 1)
    
    if use_gpu:
        model = model.cuda()
        print("Use GPU")
    else:
        print("Use CPU")
    transform = ssd_transform()
    trainset = Mydataset(data_path,transform)
    trainloader = DataLoader(
        trainset, batch_size = 5, shuffle=True
    )
    anchors = AnchorBox(use_gpu).forward()
    criterion = MultiBoxLoss(
                        use_gpu = use_gpu, 
                        iou_threshold = 0.5, 
                        variance = [0.1, 0.1],
                        class_num = cls_num+1, 
                        negpos_ratio = 3
                        )
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    for step in range(epoch):
        model.train()
        for batch_index, (img, targets) in enumerate(trainloader):
            optimizer.zero_grad()
            if use_gpu:
                img, targets = img.cuda(), targets.cuda()
            img, targets = Variable(img), Variable(targets)
            anc_box = Variable(anchors)
            loc, cls = model(img)
            loss_l, loss_c = criterion([loc, cls, anc_box], targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
        adjust_learning_rate(args.lr, optimizer, 0.95, step)


if __name__ == "__main__":
    main()