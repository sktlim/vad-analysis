# Use Miniconda as the base image
FROM continuumio/miniconda3:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    ENV_NAME=vad-analysis

# Set working directory
WORKDIR /app

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        gcc \
        g++ \
        make && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Micromamba (faster than Conda)
RUN conda install -y mamba -n base -c conda-forge && conda clean --all -y

# Copy environment.yml before running conda install (optimizes caching)
COPY environment.yml .

# Create Conda environment
RUN mamba env create -n $ENV_NAME -f environment.yml && conda clean --all -y

# Set Conda environment in PATH
ENV PATH="/opt/conda/envs/$ENV_NAME/bin:$PATH"

# Upgrade pip and install PyTorch manually to avoid long install times
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r <(grep -A 1000 "pip:" environment.yml | tail -n +2)

# Copy the rest of the application
COPY . .

# Use Conda to run Python in the correct environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "vad-analysis", "python", "main.py"]
