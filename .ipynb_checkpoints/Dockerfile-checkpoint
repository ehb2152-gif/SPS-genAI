# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./requirements.txt /code/requirements.txt

# Install all dependencies from requirements.txt
# We force torch to install the CPU-only version to keep the image small
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy your application code into the container
# This copies main.py and all helpers in the 'app' folder
COPY ./app /code/app

# Copy your trained models into the container
COPY ./checkpoints /code/checkpoints
# The `COPY ./app` command above will also copy all .pth files
# inside app folder (ebm_model.pth, generator.pth, etc.)

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# It looks for the 'app' variable inside the 'app.main' module
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]