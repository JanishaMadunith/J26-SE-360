# Build from the REPO ROOT so the shared contracts package is in the context:
#   docker build -f services/m3-schedule-optimizer/Dockerfile -t m3-schedule-optimizer .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY contracts /contracts
RUN pip install --no-cache-dir /contracts

COPY services/m3-schedule-optimizer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/m3-schedule-optimizer/ /app/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8103
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8103"]
