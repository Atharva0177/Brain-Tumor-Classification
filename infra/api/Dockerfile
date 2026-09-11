FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY app ./app
COPY worker ./worker
COPY migrations ./migrations
COPY alembic.ini ./
RUN pip install --no-cache-dir .
