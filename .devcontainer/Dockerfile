FROM python:3.11-slim-bookworm

# System dependencies required by CasADi, IPOPT, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    liblapack-dev \
    libblas-dev \
    pkg-config \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Python packages
RUN pip install \
    casadi \
    osqp \
    numpy \
    scipy \
    matplotlib \
    control \
    do-mpc \
    jupyterlab

# Disable token authentication
RUN mkdir -p /root/.jupyter && \
    echo "c.IdentityProvider.token = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py

# Working directory inside container
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888
