FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY inventory_ai ./inventory_ai
COPY scripts ./scripts
RUN pip install --no-cache-dir ".[postgres]"
EXPOSE 8000
CMD ["python", "-m", "inventory_ai", "serve", "--host", "0.0.0.0", "--port", "8000"]
