FROM python:3.12-slim

WORKDIR /app

# Set environment variable for transformers cache, which helps prevent permission errors
# HF_HOME, not TRANSFORMERS_CACHE: the latter is deprecated from transformers 4.34
# and emits a warning on every start while still needing HF_HOME set anyway.
ENV HF_HOME="/tmp/hf_home"
ENV TRANSFORMERS_CACHE="/tmp/hf_home"
ENV NUMBA_CACHE_DIR="/tmp/numba_cache"

# Copy requirements first and install packages to leverage Docker caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL your files (including run.sh, streamlit_app.py, etc.) into the container
COPY . .

# NOW that run.sh is copied, make it executable
RUN chmod +x ./run.sh

# This is the command that will run when the Space starts
CMD ["./run.sh"]