import os
import numpy
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
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
        return image, annotation,img_name
train_data = tablet_dataset(image_dir='E:\\包衣片\\20241108训练-多', annotation_dir='E:\\包衣片\\20241108训练-多', image_transform=transforms.ToTensor())
for i, (image, annotation,img_name) in enumerate(train_data):
    xml_filename = os.path.splitext(img_name)[0] + '.xml'
    xml_path = os.path.join('E:\\包衣片\\20241108训练', xml_filename)
    tree = ET.ElementTree(ET.Element("annotation"))
    root = tree.getroot()
    flags = ET.SubElement(root, "flags")
    flags.text = str(annotation)
    tree.write(xml_path)