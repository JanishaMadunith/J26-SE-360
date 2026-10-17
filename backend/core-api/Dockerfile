# Build from the REPO ROOT so the shared contracts package is available:
#   docker build -f backend/core-api/Dockerfile -t j26/core-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY contracts /contracts
RUN pip install /contracts

COPY backend/core-api/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY backend/core-api /app
RUN chmod +x /app/docker-entrypoint.sh \
    && useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); sys.exit(0)"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
