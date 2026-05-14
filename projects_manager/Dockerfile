FROM python:3.11-slim-bullseye

WORKDIR /app

# 1️⃣ Системные зависимости (slim версия требует чуть больше библиотек для работы с графикой/математикой)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libgomp1 \
    libstdc++6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Устанавливаем и фиксируем NumPy первой строкой
# Версия 1.26.4 — самая стабильная для Python 3.11 и Torch 2.2
RUN pip install --no-cache-dir numpy==1.26.4

# 3️⃣ Устанавливаем Torch и Torchvision (CPU версии)
RUN pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4️⃣ Устанавливаем остальные зависимости строго по твоим версиям
# Добавляем pandas, plotly и sqlalchemy, если они нужны бэкенду
RUN pip install --no-cache-dir \
    fastapi==0.103.2 \
    uvicorn[standard]==0.27.0 \
    requests==2.31.0 \
    jinja2==3.1.3 \
    bcrypt==4.1.2 \
    httpx==0.26.0 \
    python-multipart==0.0.7 \
    Pillow==10.2.0 \
    pandas==2.1.1 \
    plotly==5.18.0 \
    pydantic==2.4.2

COPY . .

# Отключаем создание .pyc файлов, чтобы не мусорить в контейнере
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]