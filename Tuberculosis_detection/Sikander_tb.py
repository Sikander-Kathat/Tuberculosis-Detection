# Importing libraries
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Set seeds for reproducibility
np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
torch.cuda.manual_seed_all(1234)  # multi-GPU

# Set device
cuda = True
device = torch.device("cuda" if (torch.cuda.is_available() and cuda) else "cpu")

print(device)

#zip_file_path = r'E:/downloads/archive (3).zip'

#import shutil
# Create a directory to store datasets
#os.mkdir('datasets')

# Copy the Normal and Tuberculosis folders to the datasets directory
#shutil.copytree('TB_Chest_Radiography_Database/Normal', 'datasets/normal')
#shutil.copytree('TB_Chest_Radiography_Database/Tuberculosis', 'datasets/tuberculosis')

# Copy the Normal and Tuberculosis metadata files to the datasets directory
#shutil.copy('TB_Chest_Radiography_Database/Normal.metadata.xlsx', 'datasets/normal.xlsx')
#shutil.copy('TB_Chest_Radiography_Database/Tuberculosis.metadata.xlsx', 'datasets/tuberculosis.xlsx')

# Remove the TB_Chest_Radiography_Database directory
#shutil.rmtree('TB_Chest_Radiography_Database')

# Unzipping the dataset
#with ZipFile(zip_file_path, 'r') as zip_ref:
#    zip_ref.extractall()

# Checking Excel sheets and datasets
normal_df = pd.read_excel('Tuber/normal.xlsx')
tuberculosis_df = pd.read_excel('Tuber/tuberculosis.xlsx')

print("-------------------Dataset------------------")
print("Total Normal CXR Images: ", len(normal_df))
print("Total Tuberculosis CXR Images: ", len(tuberculosis_df))

# Splitting the datasets
import glob
normal = glob.glob("Tuber/normal/*")
tb = glob.glob("Tuber/tuberculosis/*")

# Splitting in 70:15:15 ratio
train_path = normal[:2450] + tb[:490]
val_path = normal[2450:2975] + tb[490:595]
test_path = normal[2975:] + tb[595:]

# shuffling
random.shuffle(train_path)
random.shuffle(val_path)
random.shuffle(test_path)

print("-------------------Dataset------------------")
print("Total train CXR Images: ", len(train_path))
print("Total val CXR Images: ", len(val_path))
print("Total test CXR Images: ", len(test_path))

Image.open(train_path[0])

# DATA PREPROCESSING

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

# Dataset for train, valid and test
train_data = TB_CXR_Dataset(train_path, transform=data_transforms["train"])
valid_data = TB_CXR_Dataset(val_path, transform=data_transforms["others"])
test_data = TB_CXR_Dataset(test_path, transform=data_transforms["others"])

# Dataloader for train, valid, and test
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

# ...

# LeNet-5 Model
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

# Instantiate LeNet-5 model
model = LeNet5(num_classes=2)

# set device
model = model.to(device)
model

# calculating weights for each classes

def class_weight(x,total):
  return 1 - (x/total)

total = 2450 + 490
class_weights = torch.tensor([class_weight(2450,total),class_weight(490,total)])

# Define the Binary CrossEntropy with Logits loss function and Adam optimizer
loss_fn = nn.BCEWithLogitsLoss(weight = class_weights.to(device))
optimizer = optim.Adam(model.parameters(),lr=0.001)

train_losses = []
valid_losses = []
train_acc = []
valid_acc=[]
# Define a function for the training loop
def train(model, loss_fn, optimizer, num_epochs):

    for epoch in tqdm(range(num_epochs)):

        # Training
        # Set the model to training mode
        model.train()

        train_size = 0
        train_loss = 0.0
        train_accuracy = 0.0

        for batch_idx, (images, labels) in enumerate(train_dataloader):
            print(images)  # Add this line to print image paths
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, F.one_hot(labels,2).float())

            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            train_accuracy += torch.sum(preds == labels.data)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_size += images.size(0)

        # Validation
        # Set the model to evalutaion mode
        model.eval()

        valid_size = 0
        valid_loss = 0.0
        valid_accuracy = 0.0

        # Loop through the validing dataloader
        for batch_idx, (images, labels) in enumerate(valid_dataloader):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, F.one_hot(labels,2).float())

            # Compute accuracy
            _, preds = torch.max(outputs, 1)
            valid_accuracy += torch.sum(preds == labels.data)

            # Backward pass
            loss.backward()
            optimizer.step()

            valid_loss += loss.item() * images.size(0)
            valid_size += images.size(0)

        # Print training and validation statistics
        train_loss = train_loss / train_size
        train_losses.append(train_loss)
        train_accuracy = train_accuracy / train_size
        train_acc.append(train_accuracy)

        valid_loss = valid_loss / valid_size
        valid_losses.append(valid_loss)

        valid_accuracy = valid_accuracy / valid_size
        valid_acc.append(valid_accuracy)


        print(f"\nTrain Loss: {train_loss:.3f},Train Accuracy: {train_accuracy:.3f}, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.2f}")

# Train the model
train(model, loss_fn, optimizer, num_epochs=10)

"""## Evaluation"""

# plot train and test losses
plt.plot(train_losses, label='train loss')
plt.plot(valid_losses, label='validation loss')
plt.legend()
plt.show()

# plot train and test acc
train_acc_np= np.array([x.cpu().numpy() for x in train_acc])
valid_acc_np= np.array([x.cpu().numpy() for x in valid_acc])

plt.plot(train_acc_np, label='train accuracy')
plt.plot(valid_acc_np, label='validation accuracy')
plt.legend()
plt.show()

"""### Test set evaluation"""

# Evaluation on test set
y_true = []
y_hat = []

test_size = 0
test_loss = 0.0
test_accuracy = 0.0
# Loop through the testing dataloader
for batch_idx, (images, labels) in enumerate(test_dataloader):
    images, labels = images.to(device), labels.to(device)

    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(images)
    loss = loss_fn(outputs, F.one_hot(labels,2).float())

    # Compute accuracy
    _, preds = torch.max(outputs, 1)
    test_accuracy += torch.sum(preds == labels.data)

    y_true.extend(labels.detach().cpu().numpy().tolist())
    y_hat.extend(preds.detach().cpu().numpy().tolist())

    # Backward pass
    loss.backward()
    optimizer.step()

    test_loss += loss.item() * images.size(0)
    test_size += images.size(0)

from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score

metrics_acc = BinaryAccuracy()
metric_auroc = BinaryAUROC(thresholds=None)
metric_f1 = BinaryF1Score()

target = torch.tensor(y_true)
preds = torch.tensor(y_hat)

# Accuracy
acc = metrics_acc(preds, target)

# AUROC
auroc = metric_auroc(preds, target)

# F1 Score
f1 = metric_f1(preds, target)

print("Test Accuracy: ", acc)
print("Test AUROC: ", auroc)
print("Test F1: ", f1)

# Plotting epoch vs. training loss
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs. Training Loss')
plt.legend()
plt.show()

# Classification Report

from sklearn.metrics import classification_report
target_names = ['normal', 'tuberculosis']
print(classification_report(y_true, y_hat, target_names=target_names))

from sklearn.metrics import confusion_matrix

# Order of the input parameters is important:
# first param is the actual output values
# second param is what our model predicted
conf_matrix = confusion_matrix(y_true, y_hat)

conf_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Change figure size and increase dpi for better resolution
# and get reference to axes object
fig, ax = plt.subplots(figsize=(8,6), dpi=100)

# initialize using the raw 2D confusion matrix
# and output labels (in our case, it's 0 and 1)
display = ConfusionMatrixDisplay(conf_matrix, display_labels=['normal', 'tuberculosis'])

# set the plot title using the axes object
ax.set(title='Confusion Matrix for the Tuberculosis detection Model')

# show the plot.
# Pass the parameter ax to show customizations (ex. title)
display.plot(ax=ax);

"""#Saving and loading model"""

from pathlib import Path

# Create model directory path in your local folder
MODEL_PATH = Path('C:/Xampp/htdocs/Tuberculosis_detection/TuberSavedModel')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Define the model file name
MODEL_NAME = 'Lenet_5_tuberculosis_model.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the entire model
print(f'Saving the entire model to: {MODEL_SAVE_PATH}')
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# Define the path to load the model from
LOAD_PATH = "C:/Xampp/htdocs/Tuberculosis_detection/TubersavedModel/Lenet_5_tuberculosis_model.pth"

# Load the model
loaded_model = LeNet5(num_classes=2)  # Assuming LeNet5 class is defined
loaded_model.load_state_dict(torch.load(LOAD_PATH, map_location=torch.device('cpu')))
loaded_model.eval()

