FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen

COPY app/ ./app/
COPY models/ ./models/
COPY data/gold/ ./data/gold/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
