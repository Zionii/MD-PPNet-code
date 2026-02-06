import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = F.softmax(inputs, dim=1)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.softmax(inputs, dim=1)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets.float(), reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE



class EnhancedDiceLoss(nn.Module):
    def __init__(self,
                 alpha=0.5,  # 平衡因子(默认0.5)
                 gamma=1.5,  # 聚焦因子(默认1.5)
                 smooth=1e-6,  # 平滑系数(默认1e-6)
                 class_weights=None,  # 类别权重(默认None)
                 reduction='mean',  # 缩减方式(默认'mean')
                 focal=False,  # 是否启用Focal Dice(默认False)
                 tversky=True,  # 是否启用Tversky(默认False)
                 beta=0.7):  # Tversky的β参数(默认0.7)

        super(EnhancedDiceLoss4, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights
        self.reduction = reduction
        self.focal = focal
        self.tversky = tversky
        self.beta = beta

    def forward(self, inputs, targets):
   
        
        probs = F.softmax(inputs, dim=1)


        loss = 0.0
        for c in range(probs.shape[1]):
            if self.class_weights is not None:
                weight = self.class_weights[c]
            else:
                weight = 1.0

            pred = probs[:, c].contiguous().view(-1)
            target = targets[:, c].contiguous().view(-1)

            intersection = (pred * target).sum()
            pred_sum = pred.sum()
            target_sum = target.sum()

            if self.tversky:

                fp = (pred * (1 - target)).sum()
                fn = ((1 - pred) * target).sum()
                numerator = intersection + self.smooth
                denominator = numerator + self.beta * fp + (1 - self.beta) * fn + self.smooth
                dice = numerator / denominator
            else:

                dice = (2. * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)


            if self.focal:
                dice = torch.pow(1 - dice, self.gamma)

            loss += weight * (1 - dice)

        if self.reduction == 'mean':
            loss = loss / probs.shape[1]
        elif self.reduction == 'sum':
            loss = loss

        if self.alpha > 0:
            ce_loss = F.cross_entropy(inputs, torch.argmax(targets, dim=1), weight=self.class_weights)
            loss = self.alpha * loss + (1 - self.alpha) * ce_loss

        return loss


