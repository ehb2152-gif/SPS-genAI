import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from app.model import get_model # Import our factory function

# Model Loading 
MODEL_PATH = "app/ebm_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the EBM model structure from our factory
model = get_model('EBM').to(DEVICE)
model.eval()

try:
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("Loaded trained EBM model from ebm_model.pth")
except FileNotFoundError:
    print(f"WARNING: Model file not found at {MODEL_PATH}.")
    print("The /get_image_energy/ endpoint will not work.")
    model = None
except Exception as e:
    print(f"Error loading EBM model: {e}")
    model = None

# Image Transformations
# This MUST match the transformations used during EBM training
# 1. ToTensor() scales to [0, 1]
# 2. Normalize() scales from [0, 1] to [-1, 1]
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

def get_energy(image_bytes: bytes) -> dict:
    """
    Loads an image, transforms it, and passes it through the EBM
    to get a scalar energy score.
    """
    if model is None:
        return {"error": "EBM model is not loaded. Train the model first."}
        
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply transformations and add batch dimension
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get energy score
        with torch.no_grad():
            energy = model(image_tensor)
            
        return {"energy_score": energy.item()}
        
    except Exception as e:
        return {"error": str(e)}