SPS GenAI API (Assignments 1-4)

This project consolidates all models from Assignments 1 through 4 into a single, containerized FastAPI application. This serves as the final submission for Assignment 4, demonstrating a complete API deployment.

Features

Assignment 1: Word Embeddings & Text Generation (spaCy & Bigram)

Assignment 2: Image Classification (CNN on CIFAR-10)

Assignment 3: Image Generation (GAN on MNIST)

Assignment 4: Advanced Image Generation (Diffusion & EBM on CIFAR-10)

How to Run (Recommended Method: Docker)

This is the primary method for running the project, as it includes all dependencies.

1. Prerequisite

Ensure Docker Desktop is installed and running on your machine.

2. Build the Docker Image

From the root of the project directory (the one containing this README and the Dockerfile), run:

docker build -t sps-genai-api .


3. Run the Docker Container

Once the image is built, run the following command to start the API server:

# This maps port 8000 on your machine to port 8000 in the container
docker run -p 8000:8000 --name sps-api-container sps-genai-api


Note on Ports: If you get an error that port 8000 is "already allocated," it means another service (like a local Uvicorn server) is using it. You can either stop that service or run the container on a different port, like 8001:

# Example for running on port 8001
docker run -p 8001:8000 --name sps-api-container sps-genai-api


How to Test the API

Once the container is running, you can access the interactive API documentation (Swagger UI) in your browser:

Go to: http://localhost:8000/docs

(Or http://localhost:8001/docs if you used the alternative port)

From this page, you can test all the endpoints.

Key Endpoints to Test:

Assignment 1: POST /generate_text (Try with {"start_word": "the", "length": 5})

Assignment 2: POST /classify_image/ (Upload a JPG or PNG)

Assignment 3: GET /generate_digit/ (Generates an MNIST digit)

Assignment 4: GET /generate_cifar_image/ (Generates a CIFAR image)

Assignment 4: POST /get_image_energy/ (Upload an image to get its energy score)

(Alternative) Local Development Setup

If you do not wish to use Docker, you can run the project locally.

1. Setup Environment

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


2. Install Requirements

pip install -r requirements.txt


3. Download spaCy Model

This step is required for the Assignment 1 endpoints.

python -m spacy download en_core_web_md


4. Run the Server

From the project root directory:

uvicorn app.main:app --reload


The API will be available at http://127.0.0.1:8000/docs.