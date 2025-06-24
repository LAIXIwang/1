# -*- coding: utf-8 -*- #

# -----------------------------------------------------------------------
# File Name:    inference.py
# Version:      ver1_0
# Created:      2024/06/17
# Description:  本文件定义了用于在模型应用端进行推理，返回模型输出的流程
#               ★★★请在空白处填写适当的语句，将模型推理应用流程补充完整★★★
# -----------------------------------------------------------------------

import torch
from PIL import Image
from torchvision.transforms import ToTensor


def inference(image_path, model, device):
    """定义模型推理应用的流程。
    :param image_path: 输入图片的路径
    :param model: 训练好的模型
    :param device: 模型推理使用的设备，即使用哪一块CPU、GPU进行模型推理
    """
    # 将模型置为评估（测试）模式
    model.eval()

    # START----------------------------------------------------------
    # 图像预处理
    transform = Compose([
        Resize((64, 64)),  # 调整到模型需要的尺寸
        ToTensor()
    ])
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # 增加batch维度
    
    # 推理
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # 显示结果
    print(f"输入图片: {image_path}")
    print(f"预测类别: {predicted.item()}")
    # END------------------------------------------------------------


if __name__ == "__main__":
    # 指定图片路径
    image_path = "./images/test/signs/img_0006.png"

    # 加载训练好的模型
    model = torch.load('./models/model.pkl',weights_only=False)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # 显示图片，输出预测结果
    inference(image_path, model, device)
