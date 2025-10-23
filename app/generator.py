import torch
import io
from torchvision.utils import save_image
from app.model import Generator  # Imports the Generator class from app/model.py

# Model Loading 
MODEL_PATH = "generator.pth" 
DEVICE = "cpu"

# Load the model structure and set it to evaluation mode
generator = Generator().to(DEVICE)

try:
    # Load the trained weights from the .pth file
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()  # Set the model to evaluation (inference) mode
    print("Loaded trained Generator model from generator.pth")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    print("Please run 'python train.py' first to create the model file.")
    generator = None
except Exception as e:
    print(f"Error loading model: {e}")
    generator = None

def generate_digit_image():
    """
    Generates a single digit image from the GAN and returns it
    as an in-memory PNG file.
    """
    if generator is None:
        raise RuntimeError("Generator model is not loaded. Run train.py.")
    
    # 1. Create noise vector
    noise = torch.randn(1, 100, device=DEVICE) # (batch_size=1, noise_dims=100)
    
    # 2. Generate image
    with torch.no_grad(): 
        fake_image = generator(noise)
        
    # 3. Rescale image
    fake_image = (fake_image + 1) / 2
    
    # 4. Save image to in-memory buffer
    buffer = io.BytesIO()
    save_image(fake_image, buffer, format="PNG")
    
    # 5. Rewind buffer and return it
    buffer.seek(0)
    return buffer