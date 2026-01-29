# Base image with Python 3.10
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
# - ffmpeg: for video processing
# - libgl1/libglib2.0: for opencv
# - build-essential: for compiling python packages if needed
# - curl: for installing nodejs
# - libstdc++6: required by mediapipe
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libstdc++6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (LTS)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Verify versions
RUN python3 --version && node --version && npm --version

# Copy package files first for better caching
COPY package.json package-lock.json ./

# Install Node dependencies
RUN npm ci

# Copy Python requirements
COPY python/requirements.txt ./python/requirements.txt

# Install Python dependencies globally (no venv needed in Docker)
# --no-cache-dir to keep image size smaller
RUN pip install --no-cache-dir -r python/requirements.txt

# Copy the rest of the application code
COPY . .

# Build the Next.js application
# This generates the .next directory
RUN npm run build

# Expose the port (7860 for Hugging Face, or 3000 for standard)
EXPOSE 7860

# Set environment variables
ENV NODE_ENV=production
ENV PORT=7860

# CRITICAL: Tell the API to use global Python instead of creating a venv
ENV USE_GLOBAL_PYTHON=true

# CRITICAL: Add python directory to PYTHONPATH so imports work correctly
# This allows "from biomechanics import ..." to work when running track.py
ENV PYTHONPATH=/app/python:$PYTHONPATH

# Ensure Python output is not buffered (for real-time logging)
ENV PYTHONUNBUFFERED=1

# Start the application
CMD ["npm", "start"]
