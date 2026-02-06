import os
import time
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.softmax(output,dim=1).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


# 训练时的Dice指标
def dice(logits, targets, class_index):

    # if torch.is_tensor(logits):
    logits = torch.softmax(logits,dim=1)
    # if torch.is_tensor(targets):
    #     targets = targets.data.cpu().numpy()

    inter = torch.sum(logits[:, class_index, :, : ] * targets[:, class_index, :, : ])
    union = torch.sum(logits[:, class_index, :, :]) + torch.sum(targets[:, class_index, :, : ])
    dice = (2. * inter + 1) / (union + 1)
    return dice


class Denormalize(object):
    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        self._mean = -mean/std
        self._std = 1/std

    def __call__(self, tensor):
        if isinstance(tensor, np.ndarray):
            return (tensor - self._mean.reshape(-1,1,1)) / self._std.reshape(-1,1,1)
        return normalize(tensor, self._mean, self._std)
