FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       bash \
       build-essential \
       curl \
       git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv \
    && uv pip install --no-cache-dir \
       crewai \
       crewai-tools \
       pandas \
       numpy \
       matplotlib

RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

USER app

CMD ["bash"]
