import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import custom modules
from app.model import get_model
from app.trainer import train_gan

# 1. Configuration 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002 
BATCH_SIZE = 128
EPOCHS = 25  
MODEL_SAVE_PATH = "generator.pth" 

print(f"Using device: {DEVICE}")

# 2. Load Data 
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download and load the MNIST training data
dataset = datasets.MNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=transform
)

# Create the DataLoader
data_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)

print("MNIST data loaded.")

# 3. Initialize Models, Loss, and Optimizers

# Get Generator and Discriminator from model.py
generator, discriminator = get_model('GAN')

# Loss function: Binary Cross Entropy with Logits
criterion = nn.BCEWithLogitsLoss()

# Optimizers: One for each model
opt_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
opt_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

print("Models, loss, and optimizers initialized.")

# 4. Start Training 
trained_generator, _ = train_gan(
    model=(generator, discriminator),
    data_loader=data_loader,
    criterion=criterion,
    optimizer=(opt_g, opt_d),
    device=DEVICE,
    epochs=EPOCHS
)

# 5. Save the Trained Model 
torch.save(trained_generator.state_dict(), MODEL_SAVE_PATH)

print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")