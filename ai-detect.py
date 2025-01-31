import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from PIL import Image
import os
import ai_train
font_path = 'C:/Windows/Fonts/simhei.ttf'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
defect_types = {
                "黑点": 1,
                "大黑点": 2,
                "裂片": 3,
                "掉边": 4,
                "大崩盖": 5,
                "大颗粒": 6,
                "白点": 7,
                "小崩盖": 8
            }
# class BigCNN(torch.nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv=torch.nn.Sequential(
#             torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 160x160 -> 160x160
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2,2),  # 160x160 -> 80x80
#             torch.nn.Conv2d(64,128,kernel_size=3,padding=1),  # 80x80 -> 80x80
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2,2),  # 80x80 -> 40x40
#             torch.nn.Conv2d(128,256,kernel_size=3,padding=1),  # 40x40 -> 40x40
#             torch.nn.ReLU(),
#             torch.nn.MaxPool2d(2,2))  # 40x40 -> 20x20

#         self.dense=torch.nn.Sequential(
#             torch.nn.Linear(20*20*256, 1024),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(p=0.5),
#             torch.nn.Linear(1024,9))
        
#     def forward(self, x):
#         x=self.conv(x)
#         x=x.view(-1,20*20*256)
#         x=self.dense(x)
#         return x
net=ai_train.CNN().to(device)
net.load_state_dict(torch.load('CNN_model.pth'))
net.eval()
def test(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((160, 160))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = Variable(img).to(device)
    pred = net(img)
    print(pred)
    _, predicted = torch.max(pred, 1)
    # Get the predicted class
    result = predicted.item()
    defect_name = "正常" if result == 0 else [k for k, v in defect_types.items() if v == result][0]
    print(defect_name)
    # Display the image with result
    img = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    plt.title(f'File: {os.path.basename(image_path)} - Prediction: {defect_name}')
    plt.axis('off')
    plt.show()
if __name__ == '__main__':
    folder_path = 'E:/包衣片/origin'
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.bmp'):
            test(os.path.join(folder_path, filename))