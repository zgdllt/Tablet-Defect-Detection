import os
import numpy
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

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
class tablet_dataset(Dataset):
    def __init__(self, image_dir, annotation_dir, image_transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.bmp')]
        self.image_transform = image_transform
    def __len__(self):
        return len(self.image_filenames)
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('L')
        image = image.resize((160, 160))
        annotation = 0
        xml_filename = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(self.annotation_dir, xml_filename)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            flag=root.find(".//flags").text
            annotation = defect_types.get(flag, 0)  # Default to 0 if flag not found
        if self.image_transform:
            image = self.image_transform(image)
        return image, annotation
    
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(20*20*64, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 9)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 20*20*64)
        x = self.dense(x)
        return x
def train():
    net=CNN().to(device)
    train_data = tablet_dataset(image_dir='E:\\包衣片\\20241108训练', annotation_dir='E:\\包衣片\\20241108训练', image_transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True, num_workers=4, pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = net(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('Epoch %d, Loss: %.4f' % (epoch+1, total_loss))
    torch.save(net.state_dict(), 'CNN_model.pth')
if __name__ == '__main__':
    train()