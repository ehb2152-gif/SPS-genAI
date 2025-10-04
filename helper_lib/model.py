import torch.nn as nn
import torch.nn.functional as F

class FinalCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(FinalCNN, self).__init__()
        # Layer 1: Conv2D -> ReLU -> MaxPooling2D
        # Input: 64x64x3 -> Conv: 64x64x16 -> Pool: 32x32x16
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Layer 2: Conv2D -> ReLU -> MaxPooling2D
        # Input: 32x32x16 -> Conv: 32x32x32 -> Pool: 16x16x32
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # The flattened size of the volume after the second pool is 16 * 16 * 32 = 8192
        # Fully Connected Layers
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * 16 * 32, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        # Flatten the output for the FC layer
        x = x.view(-1, 16 * 16 * 32)
        x = self.fc_layer(x)
        return x

def get_model(model_name="FinalCNN", num_classes=10):
    if model_name.lower() == 'finalcnn':
        model = FinalCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Use 'FinalCNN'.")
    
    return model