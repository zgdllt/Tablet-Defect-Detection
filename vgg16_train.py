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
# Load the pre-trained VGG16 model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
class vgg16_(nn.Module):
    def __init__(self):
        super(vgg16_, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        set_parameter_requires_grad(self.features, True)
        self.classifier=nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 9)
        )
    def forward(self, x):
        x = self.features(x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
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
        image = Image.open(img_path).convert('RGB')
        annotation = 0
        xml_filename = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(self.annotation_dir, xml_filename)
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            annotation = int(root.find(".//flags").text)
            # if annotation>0:
            #     annotation=1
        if self.image_transform:
            image = self.image_transform(image)
        return image, annotation
    
# Define the image transformations
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])



# Modify the VGG16 model for binary classification
vgg16 = vgg16_().to('cuda')

# Apply Xavier initialization to the classifier layers
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

vgg16.classifier.apply(initialize_weights)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg16.parameters(), lr=0.0001)

# Initialize lists to store loss and accuracy
train_losses = []
test_accuracies = []

# Function to plot loss and accuracy
def plot_metrics(train_losses, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'b', label='Test accuracy')
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
# Training loop
if __name__ == '__main__':
    # Create the dataset and dataloader
    dataset = tablet_dataset(image_dir='E:\\包衣片\\20241108训练', annotation_dir='E:\\包衣片\\20241108训练', image_transform=image_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    testdataset = tablet_dataset(image_dir='E:\\包衣片\\origin', annotation_dir='E:\\包衣片\\origin', image_transform=image_transform)
    testloader = DataLoader(testdataset, batch_size=32, shuffle=True)
    num_epochs = 30
    for epoch in range(num_epochs):
        vgg16.train()
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            
            optimizer.zero_grad()
            outputs = vgg16(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        train_losses.append(running_loss/len(dataloader))
        vgg16.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to('cuda')
                labels = labels.to('cuda')
                outputs = vgg16(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        test_accuracies.append(accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')
        print(f'Test Accuracy: {accuracy}')
    torch.save(vgg16.state_dict(), 'vgg16_model.pth')
    plot_metrics(train_losses, test_accuracies)