# Build from the repo root: docker build -f docker/Dockerfile .
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /app/requirements.txt
# PIP_BREAK_SYSTEM_PACKAGES: honored on PEP 668 base images, ignored by older pip
RUN PIP_BREAK_SYSTEM_PACKAGES=1 pip3 install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/

WORKDIR /app

ENTRYPOINT ["python3", "main.py"]
