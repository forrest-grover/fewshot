# fewshot server: REST API + web UI. CPU by default; for an NVIDIA GPU build with
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t fewshot:gpu .
# and run with --gpus all. See "Deploying" in README.md.

FROM node:22-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FEWSHOT_HOME=/data HF_HOME=/models TORCH_HOME=/models
WORKDIR /app
# Exact versions from constraints.txt; torch without its dependencies from TORCH_INDEX (they, and a
# GPU build's CUDA libraries, come from PyPI with the rest).
COPY constraints.txt ./
RUN pip install --no-deps -c constraints.txt torch torchvision --index-url "$TORCH_INDEX"
COPY pyproject.toml README.md ./
COPY fewshot/ fewshot/
COPY --from=web /src/web/dist web/dist
# Editable, so the server finds the built UI next to the package (web/dist).
RUN pip install -c constraints.txt -e ".[siglip]" \
    && useradd --uid 1000 --create-home fewshot \
    && mkdir -p /data /models && chown fewshot /data /models
USER fewshot
VOLUME ["/data", "/models"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"
# Serving on all interfaces: set FEWSHOT_TOKEN (and FEWSHOT_ALLOWED_HOSTS to the names clients use).
CMD ["fewshot", "serve", "--host", "0.0.0.0", "--port", "8000"]
