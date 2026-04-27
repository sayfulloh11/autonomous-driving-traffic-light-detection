FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE=/workspace

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev python3-venv \
    git curl wget unzip nano \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install YOLO + common ML tools
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    matplotlib \
    tqdm \
    pyyaml \
    pandas \
    seaborn \
    jupyterlab

CMD ["bash"]
