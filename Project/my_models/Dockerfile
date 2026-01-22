FROM python:3.11-bullseye

WORKDIR /app

# 1️⃣ Системные зависимости
RUN apt-get update && apt-get install -y \
git curl wget libgomp1 libstdc++6 \
&& rm -rf /var/lib/apt/lists/*

# 2️⃣ СНАЧАЛА numpy (КРИТИЧНО)
RUN pip install --no-cache-dir numpy==1.26.4

# 3️⃣ Потом torch + torchvision
RUN pip install --no-cache-dir \
torch==2.2.0+cpu \
torchvision==0.17.0+cpu \
-f https://download.pytorch.org/whl/torch_stable.html

# 4️⃣ Остальные зависимости (БЕЗ numpy)
RUN pip install --no-cache-dir \
fastapi \
uvicorn[standard] \
requests \
pydantic \
pandas \
plotly \
matplotlib \
kagglehub \
python-multipart

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]