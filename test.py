# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    test.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了模型的测试流程
#               ★★★请在空白处填写适当的语句，将模型测试流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataset import CustomDataset


def test(dataloader, model, device):
    """定义测试流程。
    :param dataloader: 数据加载器
    :param model: 训练好的模型
    :param device: 测试使用的设备，即使用哪一块CPU、GPU进行模型测试
    """
    # 将模型置为评估（测试）模式
    model.eval()

    size = len(dataloader.dataset)  # 测试集样本总数
    correct_num = 0                 # 预测正确的样本数

    # START----------------------------------------------------------
    with torch.no_grad():
        for data in dataloader:
            images = data['image'].to(device)
            labels = data['label'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            correct_num += (predicted == labels).sum().item()
    
    accuracy = 100 * correct_num / size
    print(f"测试准确率: {accuracy:.2f}% ({correct_num}/{size})")
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 加载训练好的模型
    model = torch.load('./models/model.pkl',weights_only=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 测试数据加载器
    test_dataloader = DataLoader(CustomDataset('./images/test.txt', './images/test', ToTensor),
                                 batch_size=32)
    # 运行测试函数
    test(test_dataloader, model, device)
