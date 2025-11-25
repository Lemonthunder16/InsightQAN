# backend/Dockerfile
# Base image + small toolchain for Python libs (PyMuPDF, sentence-transformers)
FROM python:3.10-slim

# Prevent .pyc files and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install system packages required by some Python packages (PyMuPDF, ffmpeg/poppler, build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    poppler-utils \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps first (caching)
COPY requirements.txt .

# (Optional) set pip flags to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create persistent folders and set permissions (Railway will mount volumes here)
ENV CHROMA_PERSIST_DIR=/app/chroma_db
ENV ASSETS_DIR=/app/assets
RUN mkdir -p ${CHROMA_PERSIST_DIR} ${ASSETS_DIR} && \
    chown -R 1000:1000 ${CHROMA_PERSIST_DIR} ${ASSETS_DIR} || true

# Expose port (Railway will map PORT env to container)
ENV PORT=8000
EXPOSE 8000

# Use a simple, robust command (uvicorn)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
