import torch
import io
from torchvision.utils import save_image
from app.model import get_model # Import our factory function


CHECKPOINT_PATH = "app/diffusion_model.pth/diffusion_epoch_001.pth" 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the DiffusionModel wrapper (which contains the UNet)
model = get_model('DiffusionModel')
model.to(DEVICE)
model.eval()

try:
    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Load the weights for the EMA (Exponential Moving Average) network
    # The EMA network is better for generation
    model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
    
    # Load the normalization stats from the checkpoint
    model.set_normalizer(checkpoint['normalizer_mean'], checkpoint['normalizer_std'])
    
    print(f"Loaded trained Diffusion model from {CHECKPOINT_PATH}")
except FileNotFoundError:
    print(f"WARNING: Model file not found at {CHECKPOINT_PATH}.")
    print("The /generate_cifar_image/ endpoint will not work.")
    model = None
except Exception as e:
    print(f"Error loading Diffusion model: {e}")
    model = None

def generate_cifar_image():
    """
    Generates a single 32x32 CIFAR-10-like image from the Diffusion model
    and returns it as an in-memory PNG file.
    """
    if model is None:
        raise RuntimeError("Diffusion model is not loaded. Train the model first.")
        
    # 1. Generate image (returns a tensor in [0, 1] range)
    with torch.no_grad():
        # use_ema=True is important for high-quality generation
        fake_image = model.generate(num_images=1, use_ema=True)
        
    # 2. Save image to in-memory buffer
    buffer = io.BytesIO()
    save_image(fake_image, buffer, format="PNG")
    
    # 3. Rewind buffer and return it
    buffer.seek(0)
    return buffer