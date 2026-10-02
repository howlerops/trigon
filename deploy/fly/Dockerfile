# The repository's image with a bundle baked in: built from a context that has
# the repository plus ./bundle (adapter.pt and its calibrators).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[server,train]"
COPY docker/entrypoint.py /usr/local/bin/trigon-entrypoint
COPY bundle /model
RUN useradd --create-home --uid 10001 trigon && mkdir -p /cache && chown trigon /cache
USER trigon
EXPOSE 8000
ENTRYPOINT ["python", "/usr/local/bin/trigon-entrypoint"]
CMD ["trigon", "serve", "--host", "0.0.0.0", "--port", "8000"]
