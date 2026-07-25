FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --extra-index-url https://pypi.org/simple

COPY src/ src/
COPY scripts/ scripts/
COPY data/golden_set/ data/golden_set/

RUN mkdir -p data/processed data/synthetic reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "datathon_offerexp.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
