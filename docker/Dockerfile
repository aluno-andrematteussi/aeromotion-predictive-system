FROM python:3.10-slim

# Evita criação de ficheiros .pyc e força escrita imediata de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/app/models

WORKDIR /app

# Instala bibliotecas mínimas do sistema necessárias para OpenCV e compilações
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalação das dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cópia do código-fonte e artefactos treinados
COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

# Execução assíncrona com Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]