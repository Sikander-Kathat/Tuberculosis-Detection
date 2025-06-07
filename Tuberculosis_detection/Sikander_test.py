import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image


# Set seeds for reproducibility
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)  # multi-GPU

# Set device
cuda = True
device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

# Define LeNet-5 model
class LeNet5(nn.Module):
    def __init__(self, num_classes=2):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(32),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
    'others': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}

# Define custom dataset class
class TB_CXR_Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.img_paths = images
        self.img_labels = [0 if "normal" in img else 1 for img in self.img_paths]
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.img_labels[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label

# Load the dataset paths
normal = glob.glob("Tuber/normal/*")
tb = glob.glob("Tuber/tuberculosis/*")

# Define the validation dataset size
val_size = int(0.15 * len(normal))

# Initialize lists to store results
train_percentage_list = []
test_accuracy_list = []

# Loop through different training percentages
for train_percentage in range(15, 90, 15):
    train_size = int(train_percentage / 100 * len(normal))
    test_size = len(normal) - train_size - val_size
    
    # Split the dataset paths
    train_path = normal[:train_size] + tb[:train_size // 5]
    test_path = normal[train_size + val_size:] + tb[train_size // 5 + val_size:]

    # Create datasets and dataloaders
    train_data = TB_CXR_Dataset(train_path, transform=data_transforms["train"])
    test_data = TB_CXR_Dataset(test_path, transform=data_transforms["others"])

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

    # Initialize and train the model
    model = LeNet5(num_classes=2)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, loss_fn, optimizer, num_epochs=2)
    
    # Evaluate the model on the test set
    test_accuracy = calculate_accuracy(model, test_dataloader)

    # Append results to lists
    train_percentage_list.append(train_percentage)
    test_accuracy_list.append(test_accuracy)

# Plotting the results
plt.plot(train_percentage_list, test_accuracy_list)
plt.xlabel('Training Percentage')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy vs Training Percentage')
plt.show()
