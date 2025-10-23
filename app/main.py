from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import StreamingResponse  # <-- Added for Assignment 3

from app import bigram_model 
from app import image_classifier
from app import generator  

app = FastAPI(title="Assignment API")

# Pydantic models for request bodies
class WordRequest(BaseModel):
    word: str

class WordSimilarityRequest(BaseModel):
    word: str
    top_n: int

@app.get(
    "/",
    summary="Root Endpoint",
)
def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": "Welcome! This API provides image classification and word embedding services."}

@app.post(
    "/classify_image/",
    response_model=dict,
    summary="Classify an Image (Assignment 2)",
)
async def classify_image(file: UploadFile = File(...)):
    """
    Upload an image (JPG, PNG, etc.) to have the trained CNN model classify it
    into one of the 10 CIFAR-10 classes: plane, car, bird, cat, deer, dog, frog,
    horse, ship, or truck.
    """
    image_bytes = await file.read()
    predicted_class = image_classifier.get_prediction(image_bytes)
    return {"filename": file.filename, "predicted_class": predicted_class}

@app.post(
    "/get_embedding/",
    response_model=dict,
    summary="Get Word Embedding (Assignment 1)",
)
def get_embedding(request: WordRequest):
    """
    Provide a single word in the request body to receive its 300-dimensional
    vector embedding from the spaCy model.
    """
    result = bigram_model.get_embedding(request.word)
    return result
    
@app.post(
    "/similar_words",
    summary="Find Similar Words (Assignment 1)",
)
def get_similar_words(request: WordSimilarityRequest):
    """
    Provide a word and a number `top_n` to find the `n` most semantically
    similar words from a predefined vocabulary, based on the cosine similarity
    of their vector embeddings.
    """
    results = bigram_model.get_similar_words(request.word, request.top_n)
    return {"input_word": request.word, "similar_words": results}

@app.get(
    "/generate_digit/",
    response_class=StreamingResponse,
    summary="Generate a Hand-Written Digit (Assignment 3)",
)
def generate_digit():
    """
    Calls the trained GAN Generator model to produce a new 28x28
    image of a hand-written digit. Returns the image as a PNG.
    """
    image_buffer = generator.generate_digit_image()
    
    # Return the image buffer as a streaming response with the correct media type
    return StreamingResponse(image_buffer, media_type="image/png")