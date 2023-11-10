import time
import numpy as np
from data_set import *
from network import *
import matplotlib.pyplot as plt

# 使用os.environ配置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


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

def train_model():

    model, loss_function, optimizer , scheduler = model_initilize()
    train_loader, val_loader = create_dataloader()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 移动模型到cuda

    Loss = []
    Train_acc = []
    Test_acc = []
    max_acc_test = 0.80
    run_num = 0
    while (os.path.exists('run' + str(run_num))):
        run_num += 1
    os.makedirs('run'+str(run_num))
    # 训练
    print("开始训练...\n")
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
            print('Epoch-{0} learning_rater: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
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
                    torch.save(model.state_dict(), "train/run"+str(run_num)+"/best.pth")
                # 记录 Test Accuracy
                Test_acc.append(accuracy)

                print('[%d train_loss:%.3f train_accuracy:%.3f val_accuracy:%.3f]' % (epoch, running_loss / 10, train_acc ,accuracy))
                running_loss = 0.0

    print("训练完成！")
    t1 = time.time()
    training_time = t1 - t0
    print("训练用时:{}mins{}s".format(training_time//60 , training_time%60))

    # 保存模型
    print("保存模型...")
    torch.save(model.state_dict(), 'train/run'+str(run_num)+'/final.pth')
    print("模型保存成功！")
    return Loss , Train_acc , Test_acc , run_num



if __name__ == '__main__':
    Loss , Train_acc , Test_acc , run_num = train_model()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(0, epochs + 1), Loss, 'r:', label="Training Loss")
    ax1.legend(loc=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.plot(range(0, epochs + 1), Train_acc, 'g--', label="Train Accuracy")
    ax2.plot(np.arange(0 , epochs + 1 , 5), Test_acc, 'b-o', label="Test Accuracy")
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc=1)
    plt.title("Train State")
    plt.savefig('run'+str(run_num)+'/train_state.png')
    plt.show()