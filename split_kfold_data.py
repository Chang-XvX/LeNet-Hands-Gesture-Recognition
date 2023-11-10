import os
import random
from shutil import copy, rmtree
import matplotlib.pyplot as plt
import numpy as np
from pygal.style import *

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

random.seed(0)
split_rate = 0.2 # 先将test集分开
data_root = os.getcwd()
origin_finger_path = os.path.join(data_root, "finger_gesture")
generate_finger_path = os.path.join(data_root, "Kfold_dataset")
mk_file(generate_finger_path)
assert os.path.exists(origin_finger_path), "path {} does not exist".format(origin_finger_path)
assert os.path.exists(generate_finger_path), "path {} does not exist".format(generate_finger_path)


# 把手势类型写入列表 排除一些手势文件夹下出现的并目录的文件
finger_class = [cla for cla in os.listdir(origin_finger_path) if
                os.path.isdir(os.path.join(origin_finger_path, cla))]


# 建立保存测试集的文件夹
test_root = os.path.join(generate_finger_path, "test")
mk_file(test_root)
for cla in finger_class:
    mk_file(os.path.join(test_root, cla))

# 建立保存train和val集的文件夹
train_val_root = os.path.join(generate_finger_path, "train_val")
mk_file(train_val_root)
for cla in finger_class:
    mk_file(os.path.join(train_val_root, cla))

for cla in finger_class:
    cla_path = os.path.join(origin_finger_path, cla)
    images = os.listdir(cla_path)  # 将该类下的图片文件名存入列表images, 即数据集索引
    num = len(images)  # 读取图片张数
    # 随机采样验证集和测试集的索引
    test_index = random.sample(images, k=int(num * split_rate))
    for image in images:
        image_path, new_path = os.path.join(cla_path, image), ""
        if image in test_index:
            new_path = os.path.join(test_root, cla, image)
        else:
            new_path = os.path.join(train_val_root, cla, image)
        copy(image_path, new_path)

print("train_val and test datasets have been created!The rate->8:2, the data has been saved to Kfold_dataset!")
