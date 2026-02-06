# How to use it?

**训练前需将视杯视盘的image和mask放在如下文件夹下**：

`data`

​      `original_data`

​                       `image`

​                                  `fundus color images`

​                                                    `1.png`

​                                                    `2.png`

​                                                    `......`             

​                       `mask`

​                                    `Disc_Cup_Mask`

​                                                   `1.png`

​                                                   `2.png`

​                                                   `......`

## step1:

​           `python gen_txt.py`

​           `python_crop.py`

​           `python create_folder.py`（5折交叉验证）

## step2:

​            `python train.py --data REFUGE2018 --val_folder folder1 --id unet`

## step3：

​            `python test.py --data REFUGE2018 --val_folder folder1 --id unet`

## step4(CRF Post process 选用):

​           `python test_crf.py --data REFUGE2018 --val_folder folder1 --id unet`

**注：**训练期间可以通过指令 tensorboard --logdir runs 来监控损失与精度曲线