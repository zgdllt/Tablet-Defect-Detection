import glob
import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
import torch 
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Mydataset(Dataset):
    def __init__(self, images_dir, labels, annotation_dir,transform=None):
        super(Mydataset, self).__init__()
        self.images_dir = images_dir
        self.labels = labels
        self.annotation_dir = annotation_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = self.images_dir[index]
        img = Image.open(img_path)
        label = self.labels[index]

        if self.transform:
            img = self.transform(img)

        return img, label, img_path
        
def load_dataset(args):
    data = []
    labels = []

    # load images
    ng_imgs_path = glob.glob(r'ng图\*.jpg')
    ok_imgs_path = glob.glob(r'ok图\*.jpg')
    data.extend(ng_imgs_path)
    labels.extend([0] * len(ng_imgs_path))
    data.extend(ok_imgs_path)
    labels.extend([1] * len(ok_imgs_path))

    if len(data) == 0:
        raise ValueError("No images found in the specified directory.")

    train_imgs, test_imgs, train_labs, test_labs = train_test_split(data, labels, train_size=0.8)
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_dataset = Mydataset(train_imgs, train_labs, None, train_transform)
    test_dataset = Mydataset(test_imgs, test_labs, None, test_transform)

    class_counts = np.bincount(train_labs)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labs]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler, num_workers=args.workers,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,pin_memory=True)

    return train_loader, test_loader
