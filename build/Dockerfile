FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY src/api.py /app/api.py
COPY src/main.py /app/main.py
COPY src/router.py /app/router.py

WORKDIR /app

ENTRYPOINT ["python3", "main.py"]
