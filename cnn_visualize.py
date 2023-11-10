import os
import torch
from torchinfo import summary
from network import Model
import matplotlib.pyplot as plt


# 使用os.environ配置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 利用hook每次记下正向传播后特征图的样子，之后要可视化只需要读取该hook上的相对应层的信息即可
class Hook():
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self, module, fea_in, fea_out):
        print("hook working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None


# 专门用来各式各样的可视化
class Visualize():
    def __init__(self):
        self.model = Model()
        self.state_dict = torch.load('net_params.pth')
        print("模型加载...")
        self.model.load_state_dict(self.state_dict, False)
        print("模型加载成功！")

    # 模型结构和参数两可视化
    def visualize_model(self):
        print("网络可视化如下...")
        summary(self.model, (6, 3, 100, 100)) # (batch_size, in_channels, h, w)
        print("模型的各层信息及标识字典如下...")
        print(self.model)

    # kernel可视化
    def visualize_kernel(self, layer):
        # self.model.apply(init_weights)
        if layer is not None:
            conv = self.model.conv[layer]
        else:
            conv = self.model.conv[0] # 默认conv1
        kernel_set = conv.weight.detach().cpu()
        num = len(kernel_set)
        print('该层的卷积核数量为:', num)
        print('该层kernel_set形状为:', kernel_set.shape)
        for i in range(0, num):
            i_kernel = kernel_set[i]
            print(len(i_kernel))
            plt.figure(figsize=(20, 17))
            if (len(i_kernel)) >= 1:
                for idx, filer in enumerate(i_kernel):
                    # print(filer)
                    plt.subplot(1, 3, idx + 1)
                    plt.axis('off')
                    plt.imshow(filer[:,:].detach(), cmap='bwr') # bwr  绝对值越大颜色越深，正数用red负数用blue，零用white
        plt.show()

    # CNN特征图可视化
    def plot_feature(self, idx, inputs):
        hh = Hook()
        # 将hook类对象注册到要进行可视化的网络的某层中
        self.model.conv[idx].register_forward_hook(hh)
        self.model.eval()
        _ = self.model.forward(inputs)
        print(hh.module_name)
        print(hh.features_in_hook[0][0].shape)
        print(hh.features_out_hook[0].shape)

        out1 = hh.features_out_hook[0]
        total_ft = out1.shape[1]
        first_item = out1[0].cpu().clone()

        plt.figure(figsize=(20, 17))

        for ft_idx in range(total_ft):
            if ft_idx > 99:
                break
            ft = first_item[ft_idx]
            plt.subplot(10, 10, ft_idx + 1)

            plt.axis('off')
            plt.imshow(ft[:,:].detach())
        plt.show()


visual = Visualize()
inputs = torch.randn(16, 3, 100, 100)
visual.plot_feature(idx=1, inputs=inputs)
# output = visual.model.forward(inputs)


