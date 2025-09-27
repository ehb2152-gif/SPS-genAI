from collections import defaultdict, Counter
import numpy as np
import random
import re
from typing import List, Dict, Any
import spacy

# --- spaCy Integration ---
try:
    # Load the large English model
    nlp = spacy.load("en_core_web_lg")
except Exception:
    print("spaCy model failed to load. Similarity functionality unavailable.")
    nlp = None

class BigramModel:
    def __init__(self):
        self.bigram_probs = defaultdict(dict)
        self.vocab = []
        self.spacy_available = nlp is not None

    def simple_tokenizer(self, text, frequency_threshold=5):
        """Simple tokenizer that splits text into words."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not frequency_threshold:
            return tokens
        word_counts = Counter(tokens)
        filtered_tokens = [
            token for token in tokens if word_counts[token] >= frequency_threshold
        ]
        return filtered_tokens

    def train_model(self, text, frequency_threshold=None):
        """Train the bigram model on the given text."""
        words = self.simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))

        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)

        self.bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = count / unigram_counts[word1]
        
        self.vocab = list(unigram_counts.keys())
        print("Model training complete.")

    # Use spaCy vectors for similarity
    def get_similar_words(self, input_word: str, top_n: int) -> List[Dict[str, Any]]:
        """Find the top N most similar words from the vocabulary using spaCy embeddings."""
        if not self.spacy_available:
            return [{"error": "spaCy model not loaded. Similarity functionality unavailable."}]
        
        target_doc = nlp(input_word.lower())
        
        if not target_doc.has_vector:
             return [{"error": f"Word '{input_word}' does not have a vector representation."}]

        target_token = target_doc[0]

        similarities = []
        for word in self.vocab:
            if word == input_word.lower():
                continue
                
            token = nlp(word)[0]
            
            if token.has_vector and target_token.has_vector:
                 similarity_score = target_token.similarity(token)
                 similarities.append({
                    "word": word,
                    "similarity_score": similarity_score
                 })

        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return similarities[:top_n]