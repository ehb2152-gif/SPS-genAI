import spacy
import random
from typing import List, Dict, Any

# SPACY SETUP
try:
    nlp = spacy.load("en_core_web_md")
    SPACY_AVAILABLE = True
except Exception:
    print("spaCy model failed to load. NLP functionality unavailable.")
    nlp = None
    SPACY_AVAILABLE = False

# BIGRAM DATA
BIGRAM_MODEL = {
    "the": ["cat", "dog", "man", "woman", "quick", "lazy", "blue", "red"],
    "quick": ["brown", "red", "run"],
    "brown": ["fox", "dog", "bear"],
    "fox": ["jumps", "runs", "sleeps", "over"],
    "jumps": ["over", "high", "quickly"],
    "over": ["the", "a", "lazy"],
    "lazy": ["dog", "cat", "river"],
    "dog": ["barked", "slept", "ran", "jumps"],
    "cat": ["meowed", "slept", "purred"],
    "man": ["walked", "ran", "sat"],
    "woman": ["walked", "ran", "sat"],
    "sat": ["on", "under", "near"],
    "on": ["the", "a", "an"],
    "walked": ["to", "over", "slowly"],
    "ran": ["quickly", "away", "towards"],
    "a": ["blue", "red", "quick", "lazy"],
}

# FUNCTIONS 

def generate_text(start_word: str, length: int = 10) -> str:
    """Generates text using a simple bigram model."""
    current_word = start_word.lower()
    generated_words = [current_word]

    for _ in range(length):
        possible_next_words = BIGRAM_MODEL.get(current_word)
        if possible_next_words:
            next_word = random.choice(possible_next_words)
            generated_words.append(next_word)
            current_word = next_word
        else:
            break
    return " ".join(generated_words)

def get_similar_words(input_word: str, top_n: int) -> List[Dict[str, Any]]:
    """Finds similar words by searching a predefined vocabulary list."""
    if not SPACY_AVAILABLE: return [{"error": "spaCy model not loaded."}]
    
    vocab = [
        "king", "queen", "prince", "princess", "man", "woman", "boy", "girl",
        "computer", "software", "internet", "code", "apple", "google", "microsoft",
        "car", "boat", "plane", "train", "ball", "rocket", "planet", "sun", "moon", "star",
        "dog", "cat", "fish", "bird", "tree", "flower", "mountain", "river"
    ]
    
    target_doc = nlp(input_word.lower())
    if not target_doc.has_vector: return [{"error": f"Word '{input_word}' not in vocabulary."}]
    
    target_token = target_doc[0]
    similarities = []
    for word in vocab:
        if word == input_word.lower(): continue
        token = nlp(word)[0]
        if token.has_vector:
            similarity_score = target_token.similarity(token)
            similarities.append({"word": word, "similarity_score": similarity_score})
            
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    return similarities[:top_n]

def get_embedding(word: str) -> dict:
    """Returns the vector embedding for a word."""
    if not SPACY_AVAILABLE: return {"error": "spaCy model not loaded."}
    
    token = nlp(word.lower())
    if not token.has_vector: return {"error": f"Word '{word}' is out of vocabulary."}
    
    return {"word": word, "embedding": token.vector.tolist()}