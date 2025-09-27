from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI()

# Sample corpus is hardcoded but could be an external .txt file
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

# Initialize and train the model
bigram_model = BigramModel()
bigram_model.train_model(" ".join(corpus))

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int
    
class WordSimilarityRequest(BaseModel):
    word: str
    top_n: int

# Root Endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}
    
# Bigram Generation Endpoint
@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}

# spaCy Similarity Endpoint
@app.post("/similar_words", response_model=Dict[str, Any])
def get_similar_words(request: WordSimilarityRequest):
    
    if not bigram_model.spacy_available:
        return {"error": "The spaCy model failed to load. Check Docker container logs."}
        
    results = bigram_model.get_similar_words(request.word, request.top_n)
    
    return {
        "input_word": request.word,
        "similar_words": results
    }