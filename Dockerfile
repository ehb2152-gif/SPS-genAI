# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install build tools (like g++) needed to compile packages like 'thinc'
RUN apt-get update && apt-get install -y build-essential

# Copy the requirements file into the container first
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download the correct spaCy model
RUN python -m spacy download en_core_web_md

# Copy the rest of the application code, including the trained models
COPY . .

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]