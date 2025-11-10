from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import io
from app import bigram_model
from app import image_classifier
from app import generator
from app import energy_model   
from app import diffusion_model  

app = FastAPI(title="SPS Generative AI API (Assignments 1-4)")

#  Pydantic Models  
class WordRequest(BaseModel):
    word: str

class WordSimilarityRequest(BaseModel):
    word: str
    top_n: int

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

# API Endpoints 

@app.get(
    "/",
    summary="Root Endpoint",
)
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome! This API serves models from Assignments 1-4."}

# Assignment 4 Endpoints 

@app.post(
    "/get_image_energy/",
    response_model=dict,
    summary="Get Image Energy Score (Assignment 4 - EBM)",
)
async def get_image_energy(file: UploadFile = File(...)):
    """
    Upload an image (JPG, PNG, etc.) to get its 'energy' score
    from the trained Energy-Based Model (EBM).

    Lower scores suggest the image is more 'in-distribution' (looks
    more like the CIFAR-10 training data) according to the model.
    """
    image_bytes = await file.read()
    result = energy_model.get_energy(image_bytes)
    return {"filename": file.filename, **result}

@app.get(
    "/generate_cifar_image/",
    response_class=StreamingResponse,
    summary="Generate a CIFAR-10 Image (Assignment 4 - Diffusion)",
)
def generate_cifar_image():
    """
    Calls the trained Diffusion Model (UNet) to produce a new 32x32
    CIFAR-10-like image. Returns the image as a PNG.
    """
    image_buffer = diffusion_model.generate_cifar_image()
    return StreamingResponse(image_buffer, media_type="image/png")

# Assignment 3 Endpoint 

@app.get(
    "/generate_digit/",
    response_class=StreamingResponse,
    summary="Generate an MNIST Digit (Assignment 3 - GAN)",
)
def generate_digit():
    """
    Calls the trained GAN Generator model to produce a new 28x28
    image of a hand-written digit. Returns the image as a PNG.
    """
    image_buffer = generator.generate_digit_image()
    return StreamingResponse(image_buffer, media_type="image/png")

# Assignment 2 Endpoint 

@app.post(
    "/classify_image/",
    response_model=dict,
    summary="Classify an Image (Assignment 2 - CNN)",
)
async def classify_image(file: UploadFile = File(...)):
    """
    Upload an image to classify it into one of the 10 CIFAR-10 classes
    using the trained `FinalCNN` model.
    """
    image_bytes = await file.read()
    predicted_class = image_classifier.get_prediction(image_bytes)
    return {"filename": file.filename, "predicted_class": predicted_class}

# Assignment 1 Endpoints  

@app.post(
    "/get_embedding/",
    response_model=dict,
    summary="Get Word Embedding (Assignment 1)",
)
def get_embedding(request: WordRequest):
    """
    Provide a single word to receive its vector embedding.
    """
    result = bigram_model.get_embedding(request.word)
    return result
    
@app.post(
    "/similar_words",
    summary="Find Similar Words (Assignment 1)",
)
def get_similar_words(request: WordSimilarityRequest):
    """
    Provide a word and `top_n` to find the `n` most semantically
    similar words from the vocabulary.
    """
    results = bigram_model.get_similar_words(request.word, request.top_n)
    return {"input_word": request.word, "similar_words": results}

 
@app.post(
    "/generate_text",
    summary="Generate Text (Assignment 1 - Bigram)"
)
def generate_text(request: TextGenerationRequest):
    """
    Generates text using the simple bigram model.
    """
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {"generated_text": generated_text}
    except Exception as e:
        return {"error": f"Could not generate text: {e}"}




