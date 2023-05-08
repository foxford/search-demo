FROM python:3.11-slim

COPY requirements.txt main.py /app/

WORKDIR /app/
RUN pip install -U pip gunicorn==20.1.0
RUN pip install -r requirements.txt

ENTRYPOINT ["gunicorn", "-k uvicorn.workers.UvicornWorker", "--bind=0.0.0.0:8080", "--timeout=300", "main:app"]
