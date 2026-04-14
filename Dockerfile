FROM python:3.13-slim

WORKDIR /app

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

# Copy only dependency / package metadata first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install dependencies
RUN uv sync --frozen

# Copy only runtime assets needed by the app
COPY app/ ./app/
COPY models/ ./models/
COPY data/gold/ ./data/gold/

EXPOSE 8510

CMD ["uv", "run", "streamlit", "run", "app/streamlit_app.py", "--server.headless=true", "--server.port=8510", "--server.address=0.0.0.0"]