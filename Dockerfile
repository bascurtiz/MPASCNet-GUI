FROM python:3.10-slim

WORKDIR /app

# Install system audio dependencies (ffmpeg, libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code and assets
COPY . .

# Auto-download the pretrained checkpoint from Hugging Face if not already included
RUN mkdir -p checkpoint && \
    if [ ! -f checkpoint/best ]; then \
        echo "Downloading MPASCNet pretrained checkpoint..." && \
        curl -L -o checkpoint/best https://huggingface.co/Xinghour/MPASCNet/resolve/main/best; \
    fi

# Hugging Face Spaces and Web GUI default port
EXPOSE 7860
ENV PORT=7860

CMD ["python", "web_gui.py"]
