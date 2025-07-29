FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .

# remove all nvidia‑* lines and gpu runtimes
RUN sed -i '/^nvidia-/d;/onnxruntime-gpu/d' requirements.txt

# upgrade pip & add PyTorch CUDA index (optional)
RUN pip install --upgrade pip \
 && pip config set global.extra-index-url https://download.pytorch.org/whl/cu121

# install the cleaned requirements
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
