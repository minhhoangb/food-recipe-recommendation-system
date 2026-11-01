# CPU-only image for the FastAPI app. torch comes from the CPU index, so no CUDA is pulled in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# torch (CPU) first: it is the biggest wheel and the CPU index must be used for it.
# The timeouts/retries matter: the CPU wheel is ~200 MB and a slow link times out.
COPY requirements-serve.txt .
RUN pip install --timeout 120 --retries 10 torch==2.9.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --timeout 120 --retries 10 -r requirements-serve.txt

# ingredient-parser-nlp downloads one small NLTK resource on first import; the MiniLM
# weights are pre-downloaded too, so the dense backend also runs without network access.
RUN python -c "import ingredient_parser" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY api ./api
COPY src ./src
COPY configs ./configs
COPY fe ./fe

# The corpus and the bundles are mounted as volumes (see docker-compose.yml); the app
# reads them at startup and never writes to them.
# The served model is chosen by the config (serving.bundle) or by API_BUNDLE; API_BACKEND
# is left unset on purpose, so the bundle in the config wins.
ENV API_CONFIG=configs/production.yaml \
    API_REQUEST_LOG=/app/logs/requests.db

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
