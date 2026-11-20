FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FAIRJOB_MODEL_DIR=/model FAIRJOB_REDIS_URL=redis://redis:6379/0 \
    FAIRJOB_TORCH_THREADS=1 FAIRJOB_MAX_WORKERS=1 FAIRJOB_MAX_INFLIGHT=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-container.lock ./
# Explicit CPU wheels avoid installing CUDA runtime dependencies on Linux.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
    && python -m pip install -r requirements-container.lock
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-deps . && useradd --uid 10001 --create-home appuser
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "fairjob_ranker.container_entrypoint"]
