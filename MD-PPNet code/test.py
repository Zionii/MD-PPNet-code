import os
import torch
import argparse
import numpy as np
import torch. utils.data as Data
from os.path import join
from utils.common import *
from PIL import Image
from utils.metrics import *
import matplotlib
import matplotlib.pyplot as plt

from models.unet import build_unet
from Datasets.datasets import ODOC_Dataset


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (1, 512, 512)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (3, 512, 512)
    return mask


def test_refuge(test_loader, model, metrics):

    path = './Datasets/' + args.val_folder

    save_predict_path = os.path.join(args.save, args.data, args.val_folder, args.id)

    if not os.path.isdir(save_predict_path):
        os.makedirs(save_predict_path)

    with open(join(path + '/' + path.split('/')[-1] + '_' + args.train_type + '.list'),
              'r') as f:
        image_list = f.readlines()
        image_list = [item.replace('\n', '') for item in image_list]

    model.eval()

    for step, (img, lab) in enumerate(test_loader):

        image = img.float().cuda()
        target = lab.long().cuda()

        image_out = image.cpu().detach().numpy()
        image_out = np.squeeze(image_out, axis=0)
        image_out = image_out.transpose(2, 1, 0).astype(np.uint8)
        target_out = target.cpu().detach().numpy()
        target_out1 = target_out.copy()
        # print(target_out1.shape) (1, 512, 512)
        target_out = np.squeeze(target_out, axis=0)
        target_out = test_loader.dataset.decode_target(target_out).astype(np.uint8)

        metrics.reset()

        with torch.no_grad():

            output = model(image)
            # pred_y = F.softmax(output, dim=1)
            # pred_y = pred_y.argmax(1)
            # pred_y = np.array(pred_y.data.cpu()[0])
            # pred_y = np.where(pred_y == 2, 255, pred_y)
            # pred_y = np.where(pred_y == 1, 128, pred_y)
            #
            # pred_y = np.array(pred_y, dtype=np.uint8)
            pred_y = output.detach().max(dim=1)[1].cpu().numpy()
            # print(pred_y.shape) #(1, 512, 512)
            pred_y1 = pred_y.copy()
            pred_y = test_loader.dataset.decode_target(pred_y).astype(np.uint8)
            pred_y = np.squeeze(pred_y, axis=0)
            # print(pred_y.shape) #(512, 512, 3)
            # exit()

            plt.figure()
            plt.imshow(image_out)
            plt.axis('off')
            plt.imshow(pred_y, alpha=0.5)
            ax = plt.gca()
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.savefig(os.path.join(save_predict_path, "{}_overlay.jpg".format(image_list[step].split('.')[0])),
                        bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.figure()
            plt.imshow(image_out)
            plt.axis('off')
            plt.imshow(target_out, alpha=0.5)
            ax = plt.gca()
            ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
            plt.savefig(os.path.join(save_predict_path, "{}_target_overlay.jpg".format(image_list[step].split('.')[0])),
                        bbox_inches='tight', pad_inches=0)
            plt.close()

            metrics.update(target_out1, pred_y1)

            Image.fromarray(image_out).save(os.path.join(save_predict_path, "{}_image.jpg".format(image_list[step].split('.')[0])))
            Image.fromarray(target_out).save(os.path.join(save_predict_path, "{}_target.jpg".format(image_list[step].split('.')[0])))
            Image.fromarray(pred_y).save(os.path.join(save_predict_path, "{}_pred.jpg".format(image_list[step].split('.')[0])))

        score = metrics.get_results()

    return score


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='U-net add Attention mechanism for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet_fetus')
    # Path related arguments
    parser.add_argument('--root_path', default='./data/crop_data',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save', default='./result',
                        help='folder to outoput result')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_classes', default=3, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--epoch', type=int, default=300, metavar='N',
                        help='choose the specific epoch checkpoints')
    parser.add_argument('--train_type',  default='test',
                        help='Please enter test type')

    # other arguments
    parser.add_argument('--data', default='REFUGE2018', help='choose the dataset')
    # parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--att_pos', default='dec', type=str,
                        help='where attention to plug in (enc, dec, enc\&dec)')
    parser.add_argument('--view', default='axial', type=str,
                        help='use what views data to test (for fetal MRI)')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)

    # loading the dataset

    print('loading the {0} dataset ...'.format('test'))
    testset = ODOC_Dataset(dataset_folder=args.root_path, folder=args.val_folder, train_type='test')
    testloader = Data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False)
    print('Loading is done\n')

    model = build_unet(args, args.num_input, args.num_classes).cuda()

    # Load the trained best model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelname = args.ckpt + '/' + 'min_loss' + '_' + args.data + '_checkpoint.pth.tar'

    if os.path.isfile(modelname):

        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])

    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    metrics = StreamSegMetrics(args.num_classes)
    test_score = test_refuge(testloader, model, metrics=metrics)
    print(metrics.to_str(test_score))

