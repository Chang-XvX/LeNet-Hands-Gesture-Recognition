import torch.nn as nn
import torch.nn.functional as F

# 定义网络模型
class Model(nn.Module):                                                             # 3*100*100
    def __init__(self):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.25)
        self.conv1 = nn.Conv2d(3 , 6 , kernel_size=5)                               # 6*96*96
        self.conv1_BN2d = nn.BatchNorm2d(6, track_running_stats=True)               # BatchNormalization
        self.pool1 = nn.AvgPool2d(kernel_size=2 , stride=2)                         # 6*48*48
        self.conv2 = nn.Conv2d(6 , 16 , kernel_size=5)                              # 16*44*44
        self.conv2_BN2d = nn.BatchNorm2d(16, track_running_stats=True)              # BatchNormalization
        self.pool2 = nn.AvgPool2d(kernel_size=2 , stride=2)                         # 16*22*22
        self.fc1 = nn.Linear(16*22*22 , 120)
        self.fc1_BN1d = nn.BatchNorm1d(120, track_running_stats=True)               # BatchNormalization
        self.fc2 = nn.Linear(120 , 84)
        self.fc2_BN1d = nn.BatchNorm1d(84, track_running_stats=True)                # BatchNormalization
        self.fc3 = nn.Linear(84 , 10)


    def forward(self,x):
        x = F.relu(self.conv1_BN2d(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.conv2_BN2d(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(-1 , 16*22*22)
        x = self.dropout(x)
        x = F.relu(self.fc1_BN1d(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2_BN1d(self.fc2(x)))
        x = self.fc3(x)
        return x

# def init_weights(layer):
#     # 如果为卷积层，使用正态分布初始化
#     if type(layer) == nn.Conv2d:
#         print("conv2d layer initialize...")
#         nn.init.xavier_normal_(layer.weight, gain = 1.0)
#     # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
#     elif type(layer) == nn.Linear:
#         print("full connect layer initialize...")
#         nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
#         nn.init.constant_(layer.bias, 0.1)

def init_weights(layer):
    # 如果为卷积层
    if type(layer) == nn.Conv2d:
        print("conv2d layer initialize...")
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    # 如果为全连接层
    elif type(layer) == nn.Linear:
        print("full connect layer initialize...")
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.constant_(layer.bias, 0.1)