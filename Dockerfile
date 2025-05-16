# Build
# docker build --build-arg WANDB_KEY=<wandb> -t treino_sae:latest .
# Run (dgx)
# docker run --gpus '"device=0"' --rm -v /raid/aluno_gustavobarbosa/huggingface:/root/.cache/huggingface -v ./data:/app/data -v training/configs/r1-distill-llama-8b.yaml:/app/config.yaml treino_sae:latest python3 training/train_sae.py /app/config.yaml
# sudo docker run --gpus '"device=0"' --rm -v /raid/aluno_gustavobarbosa/huggingface:/root/.cache/huggingface -v ./data:/app/data -v /extraction/scripts/compute_score.sh:/app/compute_score.sh treino_sae:latest /bin/bash /app/compute_score.sh
# sudo docker run --gpus '"device=0"' --rm -v /raid/aluno_gustavobarbosa/huggingface:/root/.cache/huggingface -v $(pwd):/app treino_sae:latest /bin/bash /app/extraction/scripts/compute_score.sh

# Usa a imagem base com CUDA 12.4 e Ubuntu 22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Definir variáveis de ambiente
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Instalar dependências do Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

COPY TransformerLens /app/TransformerLens
# Instalar versão específica do TransformerLens
RUN cd TransformerLens && pip install -e .

# Instalar sae_lens e sae-dashboard

RUN pip install sae_lens==5.5.2 sae-dashboard

# Definir variável de ambiente para o Weights & Biases (WANDB)
ARG WANDB_KEY
ENV WANDB_API_KEY=$WANDB_KEY
