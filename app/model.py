import torch
import torch.nn as nn

# 1. Generator Class
class Generator(nn.Module):
    """
    GAN Generator Model
    Input: Noise vector of shape (BATCH_SIZE, 100) [cite: 5]
    Output: Image of shape (BATCH_SIZE, 1, 28, 28)
    """
    def __init__(self):
        super(Generator, self).__init__()
        
        # Fully connected layer to 7x7x128, then reshape [cite: 6]
        self.fc = nn.Linear(100, 7 * 7 * 128)
        
        # Conv Transpose2D: 128->64, kernel 4, stride 2, padding 1 [cite: 7]
        # Followed by BatchNorm2D and ReLU [cite: 8]
        self.conv_trans1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Conv Transpose2D: 64->1, kernel 4, stride 2, padding 1 [cite: 9]
        # Followed by Tanh activation [cite: 10]
        self.conv_trans2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # x shape: (BATCH_SIZE, 100)
        x = self.fc(x)
        x = x.view(-1, 128, 7, 7) # [cite: 6]
        x = self.conv_trans1(x) # Upsamples to 14x14
        x = self.conv_trans2(x) # Upsamples to 28x28
        return x

# 2. Discriminator Class 
class Discriminator(nn.Module):
    """
    GAN Discriminator Model
    Input: Image of shape (1, 28, 28) [cite: 12]
    Output: Single logit (real/fake probability)
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Conv2D: 1->64, kernel 4, stride 2, padding 1 [cite: 13]
        # Followed by LeakyReLU(0.2) [cite: 13]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Conv2D: 64->128, kernel 4, stride 2, padding 1 [cite: 14]
        # Followed by BatchNorm2D and LeakyReLU(0.2) [cite: 14]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Flatten and apply Linear layer to get a single output [cite: 15]
        self.fc = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        # x shape: (BATCH_SIZE, 1, 28, 28)
        x = self.conv1(x) # Downsamples to 14x14
        x = self.conv2(x) # Downsamples to 7x7
        x = x.view(-1, 128 * 7 * 7) # Flatten [cite: 15]
        x = self.fc(x) # Get logit
        return x

# 3. get_model Function (from Module 6 Activity)
def get_model(model_name):
    """
    Factory function to return the appropriate model.
    Based on Module 6 Activity 
    """
    
    # TODO: Add your other models like FCNN, CNN, EnhancedCNN, VAE [cite: 48]
    model = None
    
    if model_name == 'GAN': # [cite: 48]
        # For training, we need both models
        model = (Generator(), Discriminator())
    elif model_name == 'Generator':
        # For inference/generation, we only need the Generator
        model = Generator()
    elif model_name == 'Discriminator':
        model = Discriminator()
    # ... add your other elif cases for 'CNN', 'VAE', etc.
        
    if model is None:
        raise ValueError(f"Model name '{model_name}' not recognized.")
        
    return model # [cite: 49]