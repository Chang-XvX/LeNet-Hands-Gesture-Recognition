import cv2
import numpy as np
import torch
import pandas as pd
from network import Model
from torchvision.io import image
from torchvision import transforms
from data_set import create_test_dataloader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from seaborn import heatmap

run_num = 3
ifbest = 1
name = 'final'

def predict(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader = create_test_dataloader()
    right = 0.0
    sum_number = 0.0
    accuracy = 0.0
    y_true = []
    y_pred = []
    for data in dataloader:
        input = data[0].to(device)  # 移动数据到cuda
        target = data[1].to(device)  # 移动数据到cuda
        model.to(device)
        output = model.forward(input)
        predict_y = torch.max(output, dim=1)[1]
        right += torch.eq(predict_y, target).sum().item()
        sum_number += target.size(0)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(predict_y.cpu().numpy())
    accuracy = right / sum_number
    print("[Test accuracy:{}]".format(accuracy))
    cf_matrix = confusion_matrix(y_true, y_pred)
    class_names = ('0', '1', '2', '3',
                   '4', '5', '6', '7', '8', '9')

    # Create pandas dataframe
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    # Create heatmap
    heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")

    plt.title("Confusion Matrix Accuracy {:.3f}".format(accuracy)), plt.tight_layout()
    plt.ylabel("True Class"),
    plt.xlabel("Predicted Class")
    plt.savefig('train/run' + str(run_num) + '/'+ name +'_confusion_matrix.png')
    plt.show()


def predict_one(model , path ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = cv2.imread(path)
    b, g, r = cv2.split(img)  # 分别提取B、G、R通道
    img = cv2.merge([r, g, b])  # 重新组合为R、G、B
    # 设置新的分辨率
    new_resolution = (100, 100)
    # 调整图像大小
    img = cv2.resize(img, new_resolution)
    # 设置转换器
    transform = transforms.ToTensor()
    img = transform(img)
    # tensor增加一维[1,3,100,100]
    img = img.unsqueeze(0)
    input = img.float().to(device)  # 移动数据到cuda
    model.to(device)
    output = model.forward(input)
    predict_y = torch.max(output, dim=1)[1]
    img = transforms.ToPILImage()(input[0, :, :, :].to('cpu'))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
    plt.imshow(np.asarray(img))
    plt.title("预测结果:{}".format(predict_y.cpu().numpy()[0]))
    plt.show()


if __name__ == '__main__':
    # 模型加载
    model = Model().eval()
    if(ifbest):
        name = 'best'
    state_dict = torch.load('train/run'+str(run_num) +'/'+ name + '.pth')
    print("模型加载...")
    model.load_state_dict(state_dict)
    print("模型加载成功！")
    # 检测多个
    # predict(model)
    # 检测一个
    predict_one(model, "C:/Users/CXY/Desktop/Kfold_GR/6-2-2dataset/val/3/3 (2111).JPG")