FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install ffmpeg, Python 3.11, and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
