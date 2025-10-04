import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

# Define the final model architecture
class FinalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FinalCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * 16 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = x.view(-1, 16 * 16 * 32)
        x = self.fc_layer(x)
        return x

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the newly trained model
model = FinalCNN(num_classes=10)
# Load the new weights file
model.load_state_dict(torch.load('app/final_cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the CIFAR-10 class names
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_prediction(image_bytes: bytes) -> str:
    """
    Takes image bytes, processes the image with the correct transformations,
    and returns the predicted class from the FinalCNN model.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        
    predicted_class = cifar10_classes[predicted_idx.item()]
    
    return predicted_class