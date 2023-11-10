import os
import random
from shutil import copy, rmtree
import matplotlib.pyplot as plt
import numpy as np

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

random.seed(0)
split_rate = 0.4 # 先将验证集和测试集单独分出来

# 定义一下初始文件夹和生成数据文件夹
data_root = os.getcwd()
origin_finger_path = os.path.join(data_root, "hand_images")
generate_finger_path = os.path.join(data_root, "6-2-2dataset")
mk_file(generate_finger_path)
assert os.path.exists(origin_finger_path), "path {} does not exist".format(origin_finger_path)
assert os.path.exists(generate_finger_path), "path {} does not exist".format(generate_finger_path)

# 把手势类型写入列表 排除一些手势文件夹下出现的并目录的文件
finger_class = [cla for cla in os.listdir(origin_finger_path) if
                os.path.isdir(os.path.join(origin_finger_path, cla))]

# 建立保存训练集的文件夹
train_root = os.path.join(generate_finger_path, "train")
mk_file(train_root)
for cla in finger_class:
    mk_file(os.path.join(train_root, cla))

# 建立保存验证集的文件夹
val_root = os.path.join(generate_finger_path, "val")
mk_file(val_root)
for cla in finger_class:
    mk_file(os.path.join(val_root, cla))

# 建立保存测试集的文件夹
test_root = os.path.join(generate_finger_path, "test")
mk_file(test_root)
for cla in finger_class:
    mk_file(os.path.join(test_root, cla))

for cla in finger_class:
    cla_path = os.path.join(origin_finger_path, cla)
    images = os.listdir(cla_path)  # 将该类下的图片文件名存入列表images, 即数据集索引
    num = len(images)  # 读取图片张数
    # 随机采样验证集和测试集的索引
    val_test_index = random.sample(images, k=int(num * split_rate))
    for image in images:
        if image not in val_test_index:
            image_path, new_path = os.path.join(cla_path, image), ""
            new_path = os.path.join(train_root, cla, image)
            copy(image_path, new_path)
        else:
            continue
    # print(len(val_test_index))
    val_index = random.sample(val_test_index, k=int(len(val_test_index) * 0.5))
    # 在验证集和测试集中继续划分验证集
    for image in val_test_index:
        image_path, new_path = os.path.join(cla_path, image), ""
        if image in val_index:
            new_path = os.path.join(val_root, cla, image)
        else:
            new_path = os.path.join(test_root, cla, image)
        copy(image_path, new_path)
print("train, val and test datasets have been created!The rate->6:2:2")

# 可视化样本的分布
train_num, val_num, test_num = 0, 0, 0
for cla in finger_class:
    train_num += len(os.listdir(os.path.join(train_root, cla)))
    val_num += len(os.listdir(os.path.join(val_root, cla)))
    test_num += len(os.listdir(os.path.join(test_root, cla)))
print("训练集 验证集 测试集样本数量分别为:", train_num, val_num, test_num)

# 训练集中样本的分布为
cla_train_num = []
cla_val_num = []
cla_test_num = []
for cla in finger_class:
    cla_train_num += [len(os.listdir(os.path.join(train_root, cla)))]
    cla_val_num += [len(os.listdir(os.path.join(val_root, cla)))]
    cla_test_num += [len(os.listdir(os.path.join(test_root, cla)))]
index = [('num ' + x) for x in finger_class]

# fig, ax = plt.subplots()
bar_width = 0.3  # 条形宽度
index_train = np.arange(len(finger_class))  # train的横坐标
index_val = index_train + bar_width  # train条形图的横坐标
index_test = index_val + bar_width
bar1 = plt.bar(index_train, height=cla_train_num, width=bar_width, color='lightcoral', label='train')
bar2 = plt.bar(index_val, height=cla_val_num, width=bar_width, color='springgreen', label='val')
bar3 = plt.bar(index_test, height=cla_test_num, width=bar_width, color='cyan', label='test')
plt.legend()  # 显示图例
plt.xticks(index_train + bar_width * 3 / 2, index)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width*3/2 为横坐标轴刻度的位置
plt.ylabel('num')  # 纵坐标轴标题
# def auto_text(rects):
#     for rect in rects:
#         ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')
# auto_text(bar2)
# auto_text(bar3)
# plt.title('训练集、验证集、测试集样本的分布')
plt.show()







