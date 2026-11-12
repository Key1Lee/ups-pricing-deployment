FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.16
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable
ENV PATH="/app/.venv/bin:$PATH" UPS_PRICING_HOME=/app/state MPLCONFIGDIR=/tmp/matplotlib
# The real trained artifacts and SQLite directory are mounted, never baked in as placeholders.
RUN mkdir -p /app/state/data /app/state/artifacts
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=5s --start-period=20s --retries=10 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4)"
CMD ["uvicorn", "shipping.api:app", "--host", "0.0.0.0", "--port", "8000"]
