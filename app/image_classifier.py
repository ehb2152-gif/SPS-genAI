import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# 1. Define model architecture
class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Define the image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize image to the size expected by the model
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 3. Load the trained model
model = EnhancedCNN(num_classes=10)
# Note: The model is loaded onto the CPU.
model.load_state_dict(torch.load('app/enhanced_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# 4. Define the CIFAR-10 class names
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_prediction(image_bytes: bytes) -> str:
    """
    Takes image bytes, processes the image, and returns the predicted class.
    """
    # Open the image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transformations and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    # Get the class name
    predicted_class = cifar10_classes[predicted_idx.item()]
    
    return predicted_class