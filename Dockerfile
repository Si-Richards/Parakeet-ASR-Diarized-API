# GPU-friendly base (works with nvidia-container-toolkit on the host)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# System deps: ffmpeg for conversion + libs for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dirs (mount these as volumes in compose for persistence)
RUN mkdir -p /models/nemo /models/hf /tmp/asr

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
