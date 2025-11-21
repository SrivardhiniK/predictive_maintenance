# Use Python 3.10 to match TF 2.15/2.12 compatibility
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install requirements
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

# Copy project files
COPY . /app

# Expose port for Railway / containers
ENV PORT=5000
EXPOSE 5000

# Use gunicorn to serve Flask in prod
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "rul_api:app", "--workers", "1", "--threads", "2", "--timeout", "120"]
