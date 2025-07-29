FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .

# drop cuda, gpu, triton pins
RUN sed -i \
    -e '/^nvidia-/d' \
    -e '/^onnxruntime-gpu/d' \
    -e '/^triton==/d' \
    requirements.txt

RUN pip install --upgrade pip \
 && pip config set global.extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY app app
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000","--workers","1"]
