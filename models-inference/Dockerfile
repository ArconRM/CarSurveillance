FROM inference-dependencies:latest
WORKDIR /app

COPY . .

RUN ulimit -n 65535

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Stage 1: Dependencies
#FROM python:3.10 as dependencies
#WORKDIR /app
#
#RUN apt-get update && apt-get install -y \
#    gcc g++ libgomp1 libgl1 libsm6 libxext6 \
#    libxrender-dev libglib2.0-0 \
#    && rm -rf /var/lib/apt/lists/*
#
#COPY requirements.txt .
#RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
#
# Stage 2: Application
#FROM dependencies
#WORKDIR /app
#COPY . .
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
