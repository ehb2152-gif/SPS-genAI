# Use the stable Miniconda base image
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /code

# 1. Create environment and install dependencies from conda-forge and pytorch channels
RUN conda install -y \
    python=3.12 \
    fastapi \
    uvicorn \
    numpy=1.26 \
    spacy \
    pytorch \
    torchvision \
    pillow \
    -c conda-forge -c pytorch && \
    # Clean up unnecessary cached packages to keep the image small
    conda clean --all -f -y

# 2. Download and install the spaCy model 
RUN /opt/conda/bin/python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.0/en_core_web_lg-3.7.0-py3-none-any.whl

# 3. Copy project and application files
COPY pyproject.toml ./
COPY app/ ./app

# 4. Command to run the application
CMD ["/opt/conda/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]