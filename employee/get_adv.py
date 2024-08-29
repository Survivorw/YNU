from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision import transforms
from fgsm import fgsm_attack
import numpy as np


def get_adv(model, image):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    di = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

    # 看看我们有没有配置GPU，没有就是使用cpu
    device = torch.device( "cpu")

    # 设置为验证模式.
    model.eval()

    # 定义预处理转换
    preprocess = transforms.Compose([
        transforms.Resize((250, 250)),  # 调整图片尺寸为模型输入大小
        transforms.ToTensor(),  # 转换为张量
    ])

    input_tensor = preprocess(image)  # 应用预处理转换
    input_tensor = input_tensor.to(device)

    input_batch = input_tensor.unsqueeze(0)
    input_batch.requires_grad = True
    out = model(input_batch)
    pred = out.max(1, keepdim=True)[1]
    output = model(input_batch)
    pre = pred.flatten()
    sss = di[pre.item()]
    loss = F.nll_loss(output, pre)  # 计算损失函数
    loss.backward()  # 计算梯度
    data_grad = input_batch.grad
    # 调用FGSM攻击
    perturbed_data = fgsm_attack(input_batch, 0.03, data_grad)
    out = model(perturbed_data)
    pred = out.max(1, keepdim=True)[1]

    perturbed_data = perturbed_data.squeeze().detach().cpu().numpy()
    perturbed_data = np.transpose(perturbed_data, (1, 2, 0))

    plt.imsave('./app01/static/adv_image.jpg', perturbed_data)
    image = Image.open('./app01/static/adv_image.jpg').convert("RGB")
    return image, sss
