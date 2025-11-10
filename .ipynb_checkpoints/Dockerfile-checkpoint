# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install all dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Download the spaCy model data
RUN python -m spacy download en_core_web_md

# This copies main.py and all helpers/models in the 'app' folder
COPY ./app /code/app

# Copy checkpoints folder 
COPY ./checkpoints /code/checkpoints

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]