FROM python:3.11-slim-bullseye
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN apt-get update && apt-get install -y \
    git curl wget libgomp1 libstdc++6 && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch numpy
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]