FROM python:3.12-slim

RUN pip install --no-cache-dir "mlflow>=2.15" "psycopg[binary]>=3.2"

COPY infra/mlflow/entrypoint.py /entrypoint.py
ENTRYPOINT ["python", "/entrypoint.py"]
