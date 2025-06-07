import sys
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

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

# Define the path to the saved model
model_path = "C:\\xampp\\htdocs\\Tuberculosis_detection\\TuberSavedModel\\Lenet_5_tuberculosis_model.pth"

# Load the PyTorch model
model = LeNet5(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image transformation
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # Modify according to your model's input size
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale with 1 channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalize grayscale images
])

# Function to detect tuberculosis
def detect_tb(image_path):
    try:
        # Open and preprocess the image
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output = model(image)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(output, dim=1)

        # Print out probabilities for debugging
        print("Probabilities:", probabilities)

        # Get the probability of class 1 (assuming class 1 represents tuberculosis)
        tuberculosis_probability = probabilities[:, 1].item()

        # Interpret the probability based on a threshold
        threshold = 0.5  # Adjust this threshold as needed
        if tuberculosis_probability > threshold:
            return 1  # Tuberculosis detected
        else:
            return 0  # Tuberculosis not detected
    except Exception as e:
        return str(e)



# Call the function with the image path provided as an argument
# if __name__ == "__main__":
image_path = sys.argv[1]
result = detect_tb(image_path)
print(result)
sys.exit(result)
