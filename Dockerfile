# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Install the uv installer's dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest uv installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /code

# Copy the pyproject.toml and uv.lock files
COPY pyproject.toml uv.lock ./

# Install dependencies from lock file
RUN uv sync --frozen

# Install uvicorn explicitly
RUN uv pip install fastapi uvicorn numpy

# Copy the application code
COPY app/ ./app

# Command to run the application
CMD /code/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 80

