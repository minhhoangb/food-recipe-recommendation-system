# CPU-only image for the FastAPI app. torch comes from the CPU index, so no CUDA is pulled in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch (CPU) first: it is the biggest wheel and the CPU index must be used for it.
COPY requirements.txt .
RUN pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt \
    && pip install fastapi==0.162.0 uvicorn==0.54.0 httpx==0.28.1

# ingredient-parser-nlp downloads one small NLTK resource on first import; do it at build
# time so the container starts without network access.
RUN python -c "import ingredient_parser"

COPY api ./api
COPY src ./src
COPY configs ./configs
COPY fe ./fe

# The corpus and the bundles are mounted as volumes (see docker-compose.yml); the app
# reads them at startup and never writes to them.
ENV API_BACKEND=reference \
    API_CONFIG=configs/pilot.yaml \
    API_REQUEST_LOG=/app/logs/requests.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
