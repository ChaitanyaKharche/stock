FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .

# upgrade pip & add PyTorch CUDA index
RUN pip install --upgrade pip \
 && pip config set global.extra-index-url https://download.pytorch.org/whl/cu121

# now install all deps (including torch==2.5.1+cu121 etc)
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
