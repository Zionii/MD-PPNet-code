import os
import torch
import math
import visdom
import torch.utils.data as Data
import argparse
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.evaluation import AverageMeter
import pandas as pd
from tensorboardX import SummaryWriter

from Datasets.datasets import ODOC_Dataset
from models.unet import build_unet
from loss.Diceloss import DiceLoss
from utils.common import *


def train(model, loader, optimizer, loss_fn, device):
    # epoch_loss = 0.0

    # losses = AverageMeter()
    # ious = AverageMeter()
    # dices_1s = AverageMeter()
    # dices_2s = AverageMeter()

    loss_total = 0.0
    iou_total = 0.0
    train_dice_total_cup = 0.0
    train_dice_total_disc = 0.0

    model.train()
    for i,(x, y) in enumerate(loader):

        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.long)
        y = F.one_hot(y, num_classes=3)
        y = y.transpose(3, 1).contiguous()
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        iou = iou_score(y_pred, y)
        dice_1 = dice(y_pred, y, 0)
        dice_2 = dice(y_pred, y, 1)

        # losses.update(loss.item(), x.size(0))
        # ious.update(iou, x.size(0))
        # dices_1s.update(dice_1, x.size(0))
        # dices_2s.update(dice_2, x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item() * x.size(0)
        iou_total += float(iou)
        train_dice_total_cup += float(dice_1)
        train_dice_total_disc += float(dice_2)

        if i % 1 is 0:
            print('step:{},tr_loss:{:.3f},iou:{:.3f},dice_cup:{:.3f},dice_disc:{:.3f}'.format(i,loss.item(),iou,dice_1,dice_2))

    #     epoch_loss += loss.item()
    #     if i % 1 is 0:
    #         print('step:{},tr_loss:{:.3f}'.format(i,loss.item()))
    #
    # epoch_loss = epoch_loss/len(loader)
    # log = OrderedDict([
    #     ('loss', losses.avg),
    #     ('iou', ious.avg),
    #     ('dice_1', dices_1s.avg),
    #     ('dice_2', dices_2s.avg)
    # ])
    loss_ouptut_all = loss_total / loader.dataset.__len__()
    iou_total /= len(loader)
    train_dice_total_cup /= len(loader)
    train_dice_total_disc /= len(loader)
    #
    log = OrderedDict([
        ('loss', loss_ouptut_all),
        ('iou', iou_total),
        ('dice_1', train_dice_total_cup),
        ('dice_2', train_dice_total_disc)
    ])


    return log


def evaluate(model, loader, loss_fn, device):
    # epoch_loss = 0.0
    # losses = AverageMeter()
    # ious = AverageMeter()
    # dices_1s = AverageMeter()
    # dices_2s = AverageMeter()

    val_loss_total = 0.0
    val_iou_total = 0.0
    val_dice_total_cup = 0.0
    val_dice_total_disc = 0.0

    model.eval()
    with torch.no_grad():
        for i,(x, y) in enumerate(loader):

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            y = F.one_hot(y, num_classes=3)
            y = y.transpose(3, 1).contiguous()

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            iou = iou_score(y_pred, y)
            dice_1 = dice(y_pred, y, 0)
            dice_2 = dice(y_pred, y, 1)

            val_loss_total += loss.item() * x.size(0)
            val_iou_total += float(iou)
            val_dice_total_cup += float(dice_1)
            val_dice_total_disc += float(dice_2)

            if i % 1 is 0:
                print('--------------------------------------------------------------------------------------')
                print('step:{},val_loss:{:.3f},iou:{:.3f},dice_cup:{:.3f},dice_disc:{:.3f}'.format(i, loss.item(), iou,
                                                                                                  dice_1, dice_2))

            # losses.update(loss.item(), x.size(0))
            # ious.update(iou, x.size(0))
            # dices_1s.update(dice_1, x.size(0))
            # dices_2s.update(dice_2, x.size(0))


            #     epoch_loss += loss.item()
            #     if i % 1 is 0:
            #         print('step:{},tr_loss:{:.3f}'.format(i,loss.item()))
            #
            # epoch_loss = epoch_loss/len(loader)
        # log = OrderedDict([
        #     ('loss', losses.avg),
        #     ('iou', ious.avg),
        #     ('dice_1', dices_1s.avg),
        #     ('dice_2', dices_2s.avg)
        # ])

        loss_ouptut_all = val_loss_total / loader.dataset.__len__()
        val_iou_total /= len(loader)
        val_dice_total_cup /= len(loader)
        val_dice_total_disc /= len(loader)

        log = OrderedDict([
            ('loss', loss_ouptut_all),
            ('iou', val_iou_total),
            ('dice_1', val_dice_total_cup),
            ('dice_2', val_dice_total_disc)
        ])

        return log


def main(args):

    seeding(42)

    # loading the dataset
    print('loading the {0},{1},{2} dataset ...'.format('train', 'validation', 'test'))
    trainset = ODOC_Dataset(dataset_folder=args.root_path, folder=args.val_folder, train_type='train')
    validset = ODOC_Dataset(dataset_folder=args.root_path, folder=args.val_folder, train_type='validation')

    train_loader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = Data.DataLoader(dataset=validset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    print('Loading is done\n')

    # loading the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print('We can use', torch.cuda.device_count(), 'GPUs to train the network')
        model = build_unet(args, args.num_input, args.num_classes).to(device)

    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    criterion = DiceLoss()

    # Training the model
    print("Start training ...")
    # best_valid_loss = float("inf")

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'dice_1', 'dice_2', 'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2'
    ])

    best_loss = 100
    best_iou = 0
    trigger = 0
    first_time = time.time()

    writer = SummaryWriter()

    for epoch in range(args.epochs):

        print('Epoch [%d/%d]' % (epoch, args.epochs))

        train_log = train(model, train_loader, optimizer, criterion, device)
        val_log = evaluate(model, valid_loader, criterion, device)

        print('--------------------------------------------------------------------------------------')
        print('End of epoch:')
        print('loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'% (train_log['loss'], train_log['iou'],
            train_log['dice_1'], train_log['dice_2'], val_log['loss'], val_log['iou'], val_log['dice_1'], val_log['dice_2']))
        print('--------------------------------------------------------------------------------------')

        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        writer.add_scalar('Tr/Loss(end of epoch)', train_log['loss'], epoch + 1)
        writer.add_scalar('Tr/Iou(end of epoch)', train_log['iou'], epoch + 1)
        writer.add_scalar('Tr/Dice_Cup(end of epoch)', train_log['dice_1'], epoch + 1)
        writer.add_scalar('Tr/Dice_Disc(end of epoch)', train_log['dice_2'], epoch + 1)

        writer.add_scalar('Val/Loss(end of epoch)', val_log['loss'], epoch + 1)
        writer.add_scalar('Val/Iou(end of epoch)', val_log['iou'], epoch + 1)
        writer.add_scalar('Val/Dice_Cup(end of epoch)', val_log['dice_1'], epoch + 1)
        writer.add_scalar('Val/Dice_Disc(end of epoch)', val_log['dice_2'], epoch + 1)

        writer.add_scalars('Tr/Val/Loss', {'tr_loss': train_log['loss'],
                                                'val_loss': val_log['loss']}, epoch + 1)
        writer.add_scalars('Tr/Val/Iou', {'tr_iou': train_log['iou'],
                                           'val_iou': val_log['iou']}, epoch + 1)
        writer.add_scalars('Tr/Val/Dice_Cup', {'tr_dice_cup': train_log['dice_1'],
                                                'val_dice_cup': val_log['dice_1']}, epoch + 1)
        writer.add_scalars('Tr/Val/Dice_Disc', {'tr_dice_disc': train_log['dice_2'],
                                               'val_dice_disc': val_log['dice_2']}, epoch + 1)

        tmp = pd.Series([
            epoch,
            args.lr_rate,
            train_log['loss'],
            train_log['iou'],
            # train_log['dice_1'].cpu().detach().numpy(),
            # train_log['dice_2'].cpu().detach().numpy(),
            train_log['dice_1'],
            train_log['dice_2'],
            val_log['loss'],
            val_log['iou'],
            # val_log['dice_1'].cpu().detach().numpy(),
            # val_log['dice_2'].cpu().detach().numpy(),
            val_log['dice_1'],
            val_log['dice_2'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1', 'dice_2', 'val_loss', 'val_iou', 'val_dice_1', 'val_dice_2'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('{}/log.csv'.format(args.ckpt), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:

            modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'
            print('the best model will be saved at {}'.format(modelname))
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
            torch.save(state, modelname)

            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

            # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')

    # Model related arguments
    parser.add_argument('--id', default='unet', help='a name for identitying the model. Choose from the following options: Unet')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--num_input', default=3, type=int, help='number of input image for each patient')

    # Path related arguments
    parser.add_argument('--root_path', default='./data/crop_data', help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models', help='folder to output checkpoints')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 12)')
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--early-stop', default=50, type=int, metavar='N', help='early stopping (default: 30)')

    # other arguments
    parser.add_argument('--data', default='REGUGE2018', help='choose the dataset')

    # parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--val_folder', default='folder0', type=str, help='which cross validation folder')

    args = parser.parse_args()
    print("Input arguments:")

    # timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)
    print('Models are saved at %s' % (args.ckpt))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    main(args)