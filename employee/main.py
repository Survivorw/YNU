from __future__ import print_function
import torch
from PIL import Image
from get_adv import get_adv

pretrained_model = "./app01/static/model/a.pt"
model = torch.load(pretrained_model,map_location="cpu")

# 设置为验证模式. 
model.eval()

image_path = "./app01/static/img/a.jpg"
image = Image.open(image_path).convert("RGB")  # 加载图片并转换为RGB格式
adv_image = get_adv(model, image)
adv_image.show()
