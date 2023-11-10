import os
import torch
import random
from PIL import Image
from shutil import copy, rmtree
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

test_batch = 1          # 用于测试数据集，输入test_batch张
batch_size = 128         # 训练时batch
image_size = 100        # 图片大小
num_workers = 0

class GESTUREDataset(Dataset):
    '''自定义数据类'''
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        try:
            self.img = Image.open(self.images_path[item])
            if self.img.mode != 'RGB':
                raise ValueError("Image: {} isn't RGB mode.".format(self.images_path[item]))
            self.label = self.images_class[item]

            if self.transform is not None:
                self.img = self.transform(self.img)

            return self.img, self.label
        except Exception as e:
            print(f"Error reading image at index {item}: {e}")
            raise e

    # 这里定义静态方法确保不用创建实例也能执行collate_fn, 将每batch进行打包
    @staticmethod
    def collate_fn(batch):
        images , labels = tuple(zip(*batch))     # 将batch的images和labels拆出来
        images = torch.stack(images, dim = 0)   # 将图堆叠起来
        labels = torch.as_tensor(labels)        # labels也要是tensor格式
        return images, labels



def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)

def create_dataloader():
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data_train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90, expand=False),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 保证随机可复现
    random.seed(0)

    # 将数据集中10%的数据划分到验证集中
    split_rate = 0.1

    # 定义一下初始文件夹和生成数据文件夹
    data_root = os.getcwd()
    origin_finger_path = os.path.join(data_root, "hand_images")
    generate_finger_path = os.path.join(data_root, "6-2-2dataset")
    assert os.path.exists(origin_finger_path), "path {} does not exist".format(origin_finger_path)
    assert os.path.exists(generate_finger_path), "path {} does not exist".format(generate_finger_path)

    # 把手势类型写入列表 排除一些手势文件夹下出现的并目录的文件
    finger_class = [cla for cla in os.listdir(origin_finger_path) if
                    os.path.isdir(os.path.join(origin_finger_path, cla))]

    # 建立保存训练集的文件夹
    train_root = os.path.join(generate_finger_path, "train")
    # mk_file(train_root)
    # for cla in finger_class:
    #     mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(generate_finger_path, "val")
    # mk_file(val_root)
    # for cla in finger_class:
    #     mk_file(os.path.join(val_root, cla))

    # for cla in finger_class:
    #     cla_path = os.path.join(origin_finger_path, cla)
    #     images = os.listdir(cla_path)  # 将该类下的图片文件名存入列表images, 即数据集索引
    #     num = len(images)  # 读取图片张数
    #     # 随机采样验证集的索引
    #     eval_index = random.sample(images, k=int(num * split_rate))
    #     for index, image in enumerate(images):
    #         image_path, new_path = os.path.join(cla_path, image), ""
    #         if image in eval_index:
    #             new_path = os.path.join(val_root, cla, image)
    #         else:
    #             new_path = os.path.join(train_root, cla, image)
    #         copy(image_path, new_path)
    print("train and val datasets have been created!")

    # 预先创建训练图片路径和训练标签索引
    train_images_path = []
    train_images_label = []
    for cla in finger_class:
        now_dir = os.path.join(train_root, cla)
        now_images_path_list = os.listdir(now_dir)
        now_images_path_list.sort(key=lambda x: int(x[3:-5]))
        # print(now_images_path_list)
        for i in range(len(now_images_path_list)):
            now_images_path_list[i] = os.path.join(now_dir, now_images_path_list[i])
        train_images_path += now_images_path_list
        train_images_label = train_images_label + [int(cla)] * len(now_images_path_list)
    print("The list of train_images has been created!")
    # print(train_images_label)
    # print(now_images_path_list)

    # 预先创建验证图片路径和训练标签索引
    val_images_path = []
    val_images_label = []
    # print(val_images_label)
    val_root = os.path.join(generate_finger_path, "val")
    for cla in finger_class:
        now_dir = os.path.join(val_root, cla)
        now_images_path_list = os.listdir(now_dir)
        now_images_path_list.sort(key=lambda x: int(x[3:-5]))
        # print(now_images_path_list)
        for i in range(len(now_images_path_list)):
            now_images_path_list[i] = os.path.join(now_dir, now_images_path_list[i])
        val_images_path += now_images_path_list
        val_images_label = val_images_label + [int(cla)] * len(now_images_path_list)
        # print(len(now_images_path_list))
    # print(val_images_label)
    print("The list of val_images has been created!")

    # 实例化train_dataset和val_dataset
    train_dataset = GESTUREDataset(images_path=train_images_path, images_class=train_images_label,
                                   transform=data_transform)
    train_dataset_aug = GESTUREDataset(images_path=train_images_path, images_class=train_images_label,
                                       transform=data_train_transform)
    train_dataset += train_dataset_aug
    val_dataset = GESTUREDataset(images_path=val_images_path, images_class=val_images_label, transform=data_transform)

    # 实例化train_loader和val_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                               drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             drop_last=True)

    return train_loader, val_loader



def create_test_dataloader():
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
       transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 保证随机可复现
    random.seed(0)

    # 定义一下初始文件夹和生成数据文件夹
    data_root = os.getcwd()
    origin_finger_path = os.path.join(data_root, "hand_images")
    generate_finger_path = os.path.join(data_root, "6-2-2dataset")
    assert os.path.exists(origin_finger_path), "path {} does not exist".format(origin_finger_path)
    assert os.path.exists(generate_finger_path), "path {} does not exist".format(generate_finger_path)

    # 把手势类型写入列表 排除一些手势文件夹下出现的并目录的文件
    finger_class = [cla for cla in os.listdir(origin_finger_path) if
                    os.path.isdir(os.path.join(origin_finger_path, cla))]

    # 建立保存训练集的文件夹

    # 建立保存验证集的文件夹
    print("Test datasets have been created!")

    # 预先创建验证图片路径和训练标签索引
    test_images_path = []
    test_images_label = []
    # print(val_images_label)
    test_root = os.path.join(generate_finger_path, "test")
    for cla in finger_class:
        now_dir = os.path.join(test_root, cla)
        now_images_path_list = os.listdir(now_dir)
        now_images_path_list.sort(key=lambda x: int(x[3:-5]))
        # print(now_images_path_list)
        for i in range(len(now_images_path_list)):
            now_images_path_list[i] = os.path.join(now_dir, now_images_path_list[i])
        test_images_path += now_images_path_list
        test_images_label = test_images_label + [int(cla)] * len(now_images_path_list)
        # print(len(now_images_path_list))
    # print(val_images_label)
    print("The list of test_images has been created!")

    # 实例化train_dataset和val_dataset
    test_dataset = GESTUREDataset(images_path=test_images_path, images_class=test_images_label, transform=data_transform)

    # 实例化train_loader和val_loader
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True, num_workers=num_workers,
                            collate_fn=test_dataset.collate_fn, drop_last=True)

    return test_loader

def create_my_dataloader(path , label):
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
       transforms.ToTensor()
    ])
    # 保证随机可复现
    random.seed(0)
    # 实例化dataset
    test_dataset = GESTUREDataset(images_path=path, images_class=label, transform=data_transform)
    # 实例化loader
    my_loader = DataLoader(test_dataset, batch_size=1 , shuffle=True, num_workers=num_workers,
                            collate_fn=test_dataset.collate_fn, drop_last=True)

    return my_loader


def create_all_dataloader():
    data_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    data_train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90, expand=False),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    all_images_path = []
    all_images_label = []
    data_root = os.getcwd()
    all_dataset_path = os.path.join(data_root, "Kfold_dataset", "train_val")
    # 把手势类型写入列表 排除一些手势文件夹下出现的并目录的文件
    finger_class = [cla for cla in os.listdir(all_dataset_path) if
                    os.path.isdir(os.path.join(all_dataset_path, cla))]
    for cla in finger_class:
        now_dir = os.path.join(all_dataset_path, cla)
        now_images_path_list = os.listdir(now_dir)
        now_images_path_list.sort(key=lambda x: int(x[3:-5]))
        # print(now_images_path_list)
        for i in range(len(now_images_path_list)):
            now_images_path_list[i] = os.path.join(now_dir, now_images_path_list[i])
        all_images_path += now_images_path_list
        all_images_label = all_images_label + [int(cla)] * len(now_images_path_list)
    print("The list of train_val_images has been created!")
    all_dataset = GESTUREDataset(images_path=all_images_path, images_class=all_images_label,
                                  transform=data_transform)
    all_dataset_aug = GESTUREDataset(images_path=all_images_path, images_class=all_images_label,
                                  transform=data_train_transform)
    all_dataset += all_dataset_aug
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              drop_last=True)
    return all_loader