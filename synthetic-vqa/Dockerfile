ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/diffusers \
    transformers accelerate torch ftfy huggingface_hub openai datasets Pillow

USER 0
WORKDIR /workspace
COPY . /workspace

ENTRYPOINT ["python"]
