FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y \
git curl wget libgomp1 libstdc++6

# СНАЧАЛА numpy (КРИТИЧНО)
RUN pip install --no-cache-dir numpy==1.24.1

# Потом torch + torchvision
RUN pip install --no-cache-dir \
torch==2.2.0+cpu \
torchvision==0.17.0+cpu \
-f https://download.pytorch.org/whl/torch_stable.html

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]