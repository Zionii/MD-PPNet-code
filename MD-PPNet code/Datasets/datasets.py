import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset


def refuge_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


class ODOC_Dataset(Dataset):
    cmap = refuge_cmap()
    def __init__(self, dataset_folder='/crop_data',
                 folder='folder0', train_type='train'):

        self.train_type = train_type
        self.folder_file = './Datasets/' +  folder #'./Datasets/' + folder

        if self.train_type in ['train', 'validation', 'test']:
            # this is for cross validation
            with open(join(self.folder_file + '/' + self.folder_file.split('/')[-1] + '_' + self.train_type + '.list'),
                      'r') as f:
                self.image_list = f.readlines()
                # print(self.image_list)
                # exit()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            # print(self.image_list)
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            # print(self.folder)
            # exit()
            self.mask = [join(dataset_folder, 'mask', x.split('.')[0] + '_mask.png') for x in self.image_list]

        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.folder) == len(self.mask)

    def __getitem__(self, item: int):

        path_x = self.folder[item]
        path_y = self.mask[item]

        image = Image.open(path_x).convert('RGB')
        image = np.transpose(image, (2, 0, 1))
        mask = Image.open(path_y)

        image = np.array(image)
        # print(image.shape)
        # # exit()
        # image = image[1, :, :]
        image = torch.from_numpy(image)
        mask = np.array(mask)

        mask[mask == 0] = 2  #背景（黑色）
        mask[mask == 128] = 1 #视盘（灰色）
        mask[mask == 255] = 0 #视杯（白色）

        mask = torch.from_numpy(mask)

        # # return image, mask
        # if self.train_type == 'test':
        #     return image, mask, path_x,path_y
        #
        # else:
        return image,mask


    def __len__(self):
        return len(self.folder)


    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]


if __name__ == '__main__':

    # from utils.transform import ISIC2018_transform
    import torch.utils.data as Data

    root_path = '../data/crop_data/'
    val_folder = 'folder1'

    trainset = ODOC_Dataset(dataset_folder=root_path, folder=val_folder, train_type='test')

    trainloader = Data.DataLoader(dataset=trainset, batch_size=1, shuffle=True, pin_memory=True)

    for data in trainloader:
        x, y = data

        # y = to_one_hot_2d(y.long())

        print(x.shape)  # torch.Size([4, 3, 512, 512])
        print(y.shape)  # torch.Size([4, 1, 512, 512])





        y = y.squeeze(0)
        y = y.squeeze(0)
        y = y.cpu().numpy()
        print(y)

        # print(np.max(y, 0))
        # print(np.max(y, 1))
        # # exit()
        # plt.imshow(x)

        plt.imshow(y, cmap='gray')
        plt.show()



