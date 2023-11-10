import time
import numpy as np
from data_set import *
from network import *
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, KFold
import csv

# 使用os.environ配置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


lr = 1e-4                   # 学习率
decay_step = 50             # 学习率下降速率
decay_rate = 0.5            # 学习率下降率
lamda = 1e-2                # L2正则化
epochs = 20

def model_initilize(cnt_fold):
    model = Model()
    # 展示模型
    if cnt_fold == 1:
        print("网络结构如下...")
        print(model)

    model.apply(init_weights)

    # 设置损失函数
    print("设置损失函数...")
    loss_function = nn.CrossEntropyLoss().cuda()
    # 设置优化器
    print("设置优化器...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lamda)  # L2正则化
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)  # 学习率下降
    return model , loss_function , optimizer ,scheduler


def kfold_train():
    ## 将待分割文件名和类别统一写入列表
    # 定义一下初始文件夹
    data_root = os.getcwd()
    Kfold_dataset_path = os.path.join(data_root, "Kfold_dataset", "train_val")
    # 创建一下某个模型的参数保存文件夹
    Kfold_model_path = os.path.join(data_root, "Kfold_model")
    model_idx = 0
    while os.path.exists(Kfold_model_path + str(model_idx)):
        model_idx += 1
    os.makedirs(Kfold_model_path + str(model_idx))
    Kfold_model_path += str(model_idx)
    # 把手势类型写入列表 排除一些手势文件夹下出现的并目录的文件
    finger_class = [cla for cla in os.listdir(Kfold_dataset_path) if
                    os.path.isdir(os.path.join(Kfold_dataset_path, cla))]

    dataFold = []
    dataCla = []
    for cla in finger_class:
        now_dir = os.path.join(Kfold_dataset_path, cla)
        now_images_path_list = os.listdir(now_dir)
        now_images_path_list.sort(key=lambda x: int(x[3:-5]))
        for i in range(len(now_images_path_list)):
            now_images_path_list[i] = os.path.join(now_dir, now_images_path_list[i])
        # print(cla_list)
        dataFold += now_images_path_list
        dataCla += [int(cla)] * len(now_images_path_list)

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

    # 开始k折抽取
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5折 随机种子42保证可复现
    cnt_fold = 0
    avg_loss, avg_train_acc, avg_test_acc = 0, 0, 0
    # kf.split()自动进行迭代，其会切分好五折，将这五折轮流安排到test的位置上
    for train_index, test_index in kf.split(dataFold):
        cnt_fold += 1
        # 生成切分后的train和val的图片路径集合和对应的标签集合
        train_fold = torch.utils.data.dataset.Subset(dataFold, train_index)
        val_fold = torch.utils.data.dataset.Subset(dataFold, test_index)
        train_fold_cla = torch.utils.data.dataset.Subset(dataCla, train_index)
        val_fold_cla = torch.utils.data.dataset.Subset(dataCla, test_index)
        # 实例化train_dataset和val_dataset
        train_dataset = GESTUREDataset(images_path=train_fold, images_class=train_fold_cla,
                                       transform=data_transform)
        train_dataset_aug = GESTUREDataset(images_path=train_fold, images_class=train_fold_cla,
                                       transform=data_train_transform)
        val_dataset = GESTUREDataset(images_path=val_fold, images_class=val_fold_cla,
                                     transform=data_transform)
        train_dataset += train_dataset_aug
        # 实例化train_loader和val_loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True)
        Loss , Train_acc , Test_acc , run_num = train_model(Kfold_model_path, train_loader, val_loader, cnt_fold)
        rows = [Loss, Train_acc, Test_acc]
        with open(os.path.join(Kfold_model_path, 'kfold_acc.csv'), 'a+', encoding='utf8', newline='') as f:
            writer = csv.writer(f)
            if cnt_fold == 1:
                writer.writerow(['loss', 'train_acc', 'test_acc'])
            writer.writerow(rows)
        avg_loss += Loss
        avg_train_acc += Train_acc
        avg_test_acc += Test_acc

    print("5折交叉验证结束！")
    print("--------------------------------------------------------------------------------------")
    print('avg_loss:%3f' % (avg_loss/5), 'avg_train_acc:%.3f' % (avg_train_acc/5), 'avg_test_acc:%.3f' % (avg_test_acc/5))
    acc_num = 0
    while (os.path.exists('kfold_acc_' + str(acc_num) + '.csv')):
        acc_num += 1
    headers = ['avg_loss', 'avg_train_acc', 'avg_test_acc']
    rows = [avg_loss/5, avg_train_acc/5, avg_test_acc / 5]
    with open(os.path.join(Kfold_model_path, 'kfold_acc.csv'), 'a+', encoding='utf8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(rows)

def train_model(Kfold_model_path, train_loader, val_loader, cnt_fold):

    model, loss_function, optimizer , scheduler = model_initilize(cnt_fold)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda

    Loss = []
    Train_acc = []
    Test_acc = []
    max_acc_test = 0.80
    run_num = 0
    while os.path.exists(os.path.join(Kfold_model_path, 'kfold_run' + str(run_num))):
        run_num += 1
    os.makedirs(os.path.join(Kfold_model_path, 'kfold_run'+str(run_num)))
    # 训练
    print("第%d折开始训练...\n" % (cnt_fold))
    t0 = time.time()
    for epoch in range(0, epochs + 1):
        running_loss = 0.0
        train_acc = 0.0
        right = 0.0
        sum_number = 0
        model.train()
        for batch in train_loader:
            inputs = batch[0].to(device)  # 移动数据到cuda
            targets = batch[1].to(device)  # 移动数据到cuda
            # 优化器梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model.forward(inputs)
            # 计算损失函数
            loss = loss_function(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            running_loss += loss.item()
            # 计算训练准确度
            predict_y = torch.max(outputs, dim=1)[1]
            right += torch.eq(predict_y, targets).sum().item()
            sum_number += targets.size(0)
        train_acc = right / sum_number
        # 记录训练情况（Train Accuracy,Loss）
        Train_acc.append(train_acc)
        Loss.append(running_loss)
        scheduler.step()
        if epoch % 5 == 0:
            print('{2} times Epoch-{0} learning_rater: {1}'.format(epoch, optimizer.param_groups[0]['lr'], cnt_fold))
            with torch.no_grad():
                right = 0.0
                sum_number = 0
                model.eval()
                for data in val_loader:
                    test_images = data[0].to(device)  # 移动数据到cuda
                    test_labels = data[1].to(device)  # 移动数据到cuda
                    outputs = model.forward(test_images)
                    predict_y = torch.max(outputs, dim=1)[1]
                    right += torch.eq(predict_y, test_labels).sum().item()
                    sum_number += test_labels.size(0)
                accuracy = right / sum_number
                if accuracy > max_acc_test:
                    max_acc_test = accuracy
                    print("save best model")
                    # 保存模型语句
                    torch.save(model.state_dict(), os.path.join(Kfold_model_path, "kfold_run"+str(run_num)+"/best.pth"))
                # 记录 Test Accuracy
                Test_acc.append(accuracy)

                print('[times:%d epoch:%d train_loss:%.3f train_accuracy:%.3f val_accuracy:%.3f]' % (cnt_fold, epoch, running_loss / 10, train_acc ,accuracy))
                running_loss = 0.0

    print("第%d折训练完成！" % cnt_fold)
    t1 = time.time()
    training_time = t1 - t0
    print("第{}折训练用时:{}mins{}s".format(cnt_fold, training_time//60 , training_time%60))

    # 保存模型
    print("保存模型...")
    torch.save(model.state_dict(), os.path.join(Kfold_model_path, 'kfold_run'+str(run_num)+'/final.pth'))
    print("模型保存成功！")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(0, epochs + 1), Loss, 'r:', label="Training Loss")
    ax1.legend(loc=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(range(0, epochs + 1), Train_acc, 'g--', label="Train Accuracy")
    ax2.plot(np.arange(0, epochs + 1, 5), Test_acc, 'b-o', label="Test Accuracy")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc=1)
    plt.title("Train State")
    plt.savefig(os.path.join(Kfold_model_path, 'kfold_run' + str(run_num) + '/train_state.png'))
    return Loss[-1], Train_acc[-1], Test_acc[-1], run_num



if __name__ == '__main__':
    kfold_train()
    plt.show()