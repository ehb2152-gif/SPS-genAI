# SPS Generative AI Assignments API

This project contains the API and models for Assignments 1, 2, and 3, including word embeddings, image classification, and a Generative Adversarial Network (GAN).

### 1. Setup and Installation
To run this project, first create a virtual environment and install the required Python libraries.

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

# Install the required packages
```bash
pip install -r requirements.txt
```

### 2. Download spaCy Model
```bash
python -m spacy download en_core_web_md
```

### 3. Train the GAN model
```bash
python train.py
```

### 4. Run the FastAPI server
```bash
uvicorn app.main:app --reload
```

The API will be available at http://127.0.0.1:8000. You can view the automatic documentation at http://127.0.0.1:8000/docs