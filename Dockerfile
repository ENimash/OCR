FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PADDLEX_HOME=/app/.paddlex \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    HOME=/app

WORKDIR /app

FROM base AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir poetry poetry-plugin-export

COPY pyproject.toml poetry.lock* /app/
RUN poetry export -f requirements.txt --without-hashes -o /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle
RUN pip install --no-cache-dir "paddleocr[all]"

COPY src /app/src
COPY scripts/preload_models.py /app/scripts/preload_models.py

ENV CUDA_VISIBLE_DEVICES="" \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN python /app/scripts/preload_models.py

FROM base AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface
COPY --from=builder /app/.paddlex /app/.paddlex
COPY src /app/src

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLEOCR_GPU=0 \
    CUDA_VISIBLE_DEVICES=""

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=120s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"]

CMD ["uvicorn", "ocr_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
