import os


def gen_txt(txt_path, img_dir):
    f = open(txt_path, 'w')
    print(img_dir)
    for root, s_dirs, _ in os.walk(img_dir, topdown=True):  # 获取 train文件下各文件夹名称
        print(root)
        print(s_dirs)
        print(_)


        for sub_dir in s_dirs:
            i_dir = os.path.join(root, sub_dir)             # 获取各类的文件夹 绝对路径
            # 获取类别文件夹下所有图片的路径
            img_list = os.listdir(i_dir)
            for i in range(len(img_list)):
                if img_list[i].startswith('.'):
                    continue
                # label = img_list[i][0]
                img_path = os.path.join(i_dir, img_list[i])
                line = img_path + '\n'
                f.write(line)
    f.close()


if __name__ == '__main__':

    image_txt_path = os.path.join("../data/crop_data", "images.txt")
    mask_txt_path = os.path.join("../data/crop_data", "masks.txt")

    valid_dir = os.path.join("../data/original_data", "image")
    valid_mask_dir = os.path.join("../data/original_data", "mask")

    gen_txt(image_txt_path, valid_dir)
    gen_txt(mask_txt_path, valid_mask_dir)
