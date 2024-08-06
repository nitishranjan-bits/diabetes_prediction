FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY mlruns /app/mlruns
COPY scaler.npy /app/scaler.npy

CMD ["python", "src/app.py"]
