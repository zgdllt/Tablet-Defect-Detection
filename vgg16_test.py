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
import time
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
image_dir = 'testdata'
vgg16=vgg16_train.vgg16_()
vgg16.load_state_dict(torch.load(model_path))
vgg16.eval()
filenames=[f for f in os.listdir(image_dir) if f.endswith('.bmp')]
for imagename in filenames:
    image_path = os.path.join(image_dir, imagename)
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    start_time = time.time()
    output = vgg16(image)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.4f} seconds")
    _, predicted = torch.max(output, 1)
    predicted_defect = defect_types[predicted.item()]
    print(f"Predicted defect type: {predicted_defect}")
    plt.imshow(image.squeeze().permute(1, 2, 0))
    plt.title(f"Predicted defect type: {predicted_defect}", fontproperties='SimHei')
    plt.xlabel(f"Inference time: {inference_time:.4f} seconds")
    plt.show()