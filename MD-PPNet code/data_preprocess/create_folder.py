import os
import numpy
from random import shuffle

PATH = '../data/crop_data/image'
SAVE_PATH = '../Datasets'


def create_5_floder(folder, save_foler):
    file_list = os.listdir(folder)
    print(file_list)

    shuffle(file_list)

    for i in range(5):
        if i != 0:
            pre_test_list = file_list[0:i*50]
        else:
            pre_test_list = []
        test_list = file_list[i*50:(i+1)*50]

        if i < 4:
            valid_list = file_list[(i+1)*50:(i+1)*50+150]
            train_list = file_list[(i+1)*50+150:] + pre_test_list
        else:
            valid_list = file_list[-50:] + file_list[:100]
            train_list = file_list[100:i*175]

        if not os.path.isdir(save_foler + '/folder'+str(i+1)):
            os.makedirs(save_foler + '/folder'+str(i+1))

        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_train.list'), train_list)
        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_validation.list'), valid_list)
        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_test.list'), test_list)


def text_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))


if __name__ == "__main__":
    create_5_floder(PATH, SAVE_PATH)
