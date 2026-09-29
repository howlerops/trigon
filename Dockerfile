# The Trigon gateway, CPU build: `/v1/decide`, `/compat`, `/healthz`.
#
#   docker run -p 8000:8000 ghcr.io/howlerops/trigon            # lexical floor, no weights
#   docker run -p 8000:8000 \
#     -e TRIGON_BUNDLE_URL=https://github.com/howlerops/trigon/releases/download/banking77-qwen15b-v2/banking77-qwen15b-v2.tar.gz \
#     -e TRIGON_BUNDLE_SHA256=$(cut -d' ' -f1 releases/banking77-qwen15b-v2/BUNDLE.sha256) \
#     -v trigon-cache:/cache ghcr.io/howlerops/trigon          # the certified model
#
# Configured by the same TRIGON_* variables `trigon serve` reads
# (src/trigon/server/config.py). No weights are baked in: an image is code, a
# model is data with its own checksum, and a model is fetched and verified at
# start (docker/entrypoint.py) or mounted.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRIGON_WEIGHTS_CACHE=/cache/backbones

WORKDIR /app
# CPU torch from its own index first, so the `train` extra does not pull the
# CUDA wheel -- gigabytes an image without a GPU would never use.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,train]"
COPY docker/entrypoint.py /usr/local/bin/trigon-entrypoint

RUN useradd --create-home --uid 10001 trigon && mkdir -p /cache /model \
    && chown trigon /cache /model
USER trigon
VOLUME ["/cache"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
ENTRYPOINT ["python", "/usr/local/bin/trigon-entrypoint"]
CMD ["trigon", "serve", "--host", "0.0.0.0", "--port", "8000"]
