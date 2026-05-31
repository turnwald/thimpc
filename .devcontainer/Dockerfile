FROM python:3.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/workspace

# System dependencies required by CasADi, IPOPT, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Python packages shared by local devcontainers and GitLab validation
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Working directory inside container
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888
