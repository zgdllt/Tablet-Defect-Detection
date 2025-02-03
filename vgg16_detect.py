import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.models as models
import matplotlib.pyplot as plt
import vgg16_train
defect_types = {
                0: "无缺陷",
                1: "黑点",
                2: "大黑点",
                3: "裂片",
                4: "掉边",
                5: "大崩盖",
                6: "大颗粒",
                7: "白点",
                8: "小崩盖"
            }
model_path = 'vgg16_model.pth'
image_dir = 'test.png'
vgg16=vgg16_train.vgg16_()
vgg16.load_state_dict(torch.load(model_path))
vgg16.eval()
image = Image.open(image_dir).convert('RGB')
image = transforms.ToTensor()(image)
image = image.unsqueeze(0)
output = vgg16(image)
_, predicted = torch.max(output, 1)
predicted_defect = defect_types[predicted.item()]
print(f"Predicted defect type: {predicted_defect}")