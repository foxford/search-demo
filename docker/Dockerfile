FROM python:3.11-slim

WORKDIR "/app"
COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py

RUN pip install -U pip gunicorn==20.1.0
RUN pip install -r requirements.txt

ENTRYPOINT ["gunicorn", "-k uvicorn.workers.UvicornWorker", "--bind=0.0.0.0:8080", "--timeout=300", "main:app"]
