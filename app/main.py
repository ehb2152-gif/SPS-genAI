from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
from app.bigram_model import BigramModel
from app.image_classifier import get_prediction

app = FastAPI(title="Multi-Model API")

# Bigram Model Initialization
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel()
bigram_model.train_model(" ".join(corpus))

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class WordSimilarityRequest(BaseModel):
    word: str
    top_n: int

# API Endpoints

@app.get("/")
def read_root():
    return {"message": "Welcome to the Multi-Model API! Visit /docs for details."}

# Image Classification Endpoint
@app.post("/classify_image/", response_model=dict)
async def classify_image(file: UploadFile = File(...)):
    """
    Accepts an image file, classifies it using the trained CNN,
    and returns the predicted class.
    """
    # Read the image bytes from the uploaded file
    image_bytes = await file.read()
    
    # Get the prediction from the image_classifier module
    predicted_class = get_prediction(image_bytes)
    
    return {"filename": file.filename, "predicted_class": predicted_class}


# Original Bigram Model Endpoints
@app.post("/generate_text")
def generate_text(request: TextGenerationRequest):
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {"generated_text": generated_text}
    except AttributeError:
        return {"error": "generate_text method not found in BigramModel."}

@app.post("/similar_words")
def get_similar_words(request: WordSimilarityRequest):
    if not bigram_model.spacy_available:
        return {"error": "spaCy model not available."}
    results = bigram_model.get_similar_words(request.word, request.top_n)
    return {"input_word": request.word, "similar_words": results}