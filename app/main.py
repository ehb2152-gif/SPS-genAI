from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
import logging

# Configure logging for easier debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import model modules
# Assumes these modules handle their own model loading on import
try:
    from app import bigram_model
    from app import image_classifier
    from app import generator
    from app import energy_model
    from app import diffusion_model
    logger.info("All model modules imported successfully.")
except ImportError as e:
    logger.error(f"Failed to import a model module: {e}")

# API SETUP & METADATA
tags_metadata = [
    {"name": "General", "description": "Health checks and root endpoints."},
    {"name": "Assignment 1", "description": "NLP: Word Embeddings and Bigram Text Generation."},
    {"name": "Assignment 2", "description": "CV: CNN Image Classification (CIFAR-10)."},
    {"name": "Assignment 3", "description": "GenAI: GAN Digit Generation (MNIST)."},
    {"name": "Assignment 4", "description": "GenAI: Diffusion and Energy-Based Models (CIFAR-10)."},
]

app = FastAPI(
    title="SPS Generative AI API",
    description="Combined API for Deep Learning Assignments 1-4 demonstrating embeddings, CNNs, GANs, Diffusion, and EBMs.",
    version="1.0.0",
    openapi_tags=tags_metadata
)

# PYDANTIC MODELS 
class WordRequest(BaseModel):
    word: str

class WordSimilarityRequest(BaseModel):
    word: str
    top_n: int = 5  # Added default value

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 20 # Added default value

#  HELPER FUNCTIONS  
def validate_image(file: UploadFile):
    """Ensures uploaded files are images."""
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only JPEG and PNG images are supported."
        )

# GENERAL ENDPOINTS 
@app.get("/", tags=["General"], summary="Root Health Check")
def read_root():
    """Confirms the API is running and accessible."""
    return {"status": "online", "message": "Welcome to the SPS GenAI Combined API (Assignments 1-4)"}

# ASSIGNMENT 1: NLP & Embeddings
@app.post("/get_embedding/", tags=["Assignment 1"], response_model=dict, summary="Get Word Embedding")
def get_embedding(request: WordRequest):
    """Retrieves the vector embedding for a given word."""
    try:
        result = bigram_model.get_embedding(request.word.lower())
        return result
    except Exception as e:
        logger.error(f"Embedding error for word '{request.word}': {e}")
        raise HTTPException(status_code=404, detail=f"Word not found or model error: {e}")

@app.post("/similar_words", tags=["Assignment 1"], summary="Find Semantically Similar Words")
def get_similar_words(request: WordSimilarityRequest):
    """Finds the top-N most similar words in the vocabulary."""
    try:
        results = bigram_model.get_similar_words(request.word.lower(), request.top_n)
        return {"input_word": request.word, "similar_words": results}
    except Exception as e:
        logger.error(f"Similarity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_text", tags=["Assignment 1"], summary="Generate Text (Bigram)")
def generate_text(request: TextGenerationRequest):
    """Generates text using the simple bigram model starting from a seed word."""
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {"generated_text": generated_text}
    except Exception as e:
        logger.error(f"Text generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Could not generate text: {e}")

# ASSIGNMENT 2: CNN Classifier 
@app.post("/classify_image/", tags=["Assignment 2"], response_model=dict, summary="Classify CIFAR-10 Image")
async def classify_image(file: UploadFile = File(...)):
    """Uploads an image and returns the predicted CIFAR-10 class."""
    validate_image(file)
    try:
        image_bytes = await file.read()
        predicted_class = image_classifier.get_prediction(image_bytes)
        return {"filename": file.filename, "predicted_class": predicted_class}
    except Exception as e:
        logger.error(f"CNN Classification error: {e}")
        raise HTTPException(status_code=500, detail="Model failed to classify image.")

# ASSIGNMENT 3: GAN Generator  
@app.get("/generate_digit/", tags=["Assignment 3"], summary="Generate MNIST Digit (GAN)")
def generate_digit():
    """Generates a new hand-written digit image using GAN."""
    try:
        image_buffer = generator.generate_digit_image()
        # Reset buffer pointer to the beginning before streaming
        if isinstance(image_buffer, io.BytesIO):
             image_buffer.seek(0)
        return StreamingResponse(image_buffer, media_type="image/png")
    except Exception as e:
        logger.error(f"GAN generation error: {e}")
        raise HTTPException(status_code=500, detail="GAN model failed to generate image.")

# ASSIGNMENT 4: Diffusion & Energy Models 
@app.post("/get_image_energy/", tags=["Assignment 4"], response_model=dict, summary="Calculate Image Energy (EBM)")
async def get_image_energy(file: UploadFile = File(...)):
    """Calculates the energy score of an uploaded image using EBM."""
    validate_image(file)
    try:
        image_bytes = await file.read()
        result = energy_model.get_energy(image_bytes)
        return {"filename": file.filename, **result}
    except Exception as e:
        logger.error(f"EBM error: {e}")
        raise HTTPException(status_code=500, detail="Energy model failed to process image.")

@app.get("/generate_cifar_image/", tags=["Assignment 4"], summary="Generate CIFAR Image (Diffusion)")
def generate_cifar_image():
    """Generates a new CIFAR-like image using the Diffusion model."""
    try:
        image_buffer = diffusion_model.generate_cifar_image()
        if isinstance(image_buffer, io.BytesIO):
             image_buffer.seek(0)
        return StreamingResponse(image_buffer, media_type="image/png")
    except Exception as e:
        logger.error(f"Diffusion generation error: {e}")
        raise HTTPException(status_code=500, detail="Diffusion model failed to generate image.")