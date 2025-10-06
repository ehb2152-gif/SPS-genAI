import spacy
from typing import List, Dict, Any

try:
    nlp = spacy.load("en_core_web_lg")
    SPACY_AVAILABLE = True
except Exception:
    print("spaCy model failed to load. NLP functionality unavailable.")
    nlp = None
    SPACY_AVAILABLE = False

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