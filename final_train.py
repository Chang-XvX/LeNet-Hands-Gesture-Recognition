import time
import numpy as np
from data_set import *
from network import *
import matplotlib.pyplot as plt
import csv

# 使用os.environ配置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


lr = 1e-4                   # 学习率
decay_step = 50             # 学习率下降速率
decay_rate = 0.5            # 学习率下降率
lamda = 1e-2                # L2正则化
epochs = 100

def model_initilize():
    model = Model()
    # 展示模型
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

def train_model(final_model_path):

    model, loss_function, optimizer , scheduler = model_initilize()
    all_loader = create_dataloader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda
    Loss = []
    Train_acc = []
    max_acc_test = 0.80
    # 训练
    print("开始训练...\n")
    t0 = time.time()
    for epoch in range(0, epochs + 1):
        running_loss = 0.0
        train_acc = 0.0
        right = 0.0
        sum_number = 0
        model.train()
        for batch in all_loader:
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
            print('Epoch-{0} learning_rate: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            print('[%d train_loss:%.3f train_accuracy:%.3f]' % (epoch, running_loss / 10, train_acc))
            running_loss = 0.0

    print("训练完成！")
    t1 = time.time()
    training_time = t1 - t0
    print("训练用时:{}mins{}s".format(training_time//60 , training_time%60))

    # 保存模型
    print("保存模型...")
    torch.save(model.state_dict(), os.path.join(final_model_path, 'final.pth'))
    print("模型保存成功！")
    return Loss , Train_acc



if __name__ == '__main__':
    # 定义一下初始文件夹
    data_root = os.getcwd()
    # 创建一下某个模型的参数保存文件夹
    final_model_path = os.path.join(data_root, "final_model")
    model_idx = 0
    while os.path.exists(final_model_path + str(model_idx)):
        model_idx += 1
    os.makedirs(final_model_path + str(model_idx))
    final_model_path += str(model_idx)

    Loss , Train_acc = train_model(final_model_path)
    with open(os.path.join(final_model_path, 'final_acc.csv'), 'a+', encoding='utf8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['final_loss', 'final_train_acc'])
        writer.writerow([Loss[-1], Train_acc[-1]])
    print("--------------------------------------------------------------------------------------")
    print('final_loss:%3f' % (Loss[-1]), 'final_train_acc:%.3f' % (Train_acc[-1]))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(0, epochs + 1), Loss, 'r:', label="Training Loss")
    ax1.legend(loc=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(range(0, epochs + 1), Train_acc, 'g--', label="Train Accuracy")
    # ax2.plot(np.arange(0 , epochs + 1 , 5), Test_acc, 'b-o', label="Test Accuracy")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc=1)
    plt.title("Train State")
    plt.savefig(os.path.join(final_model_path, 'train_state.png'))
    plt.show()