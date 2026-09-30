FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

WORKDIR /app

ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 HF_HOME=/opt/hf-cache
ARG RERANKER_MODEL

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
RUN uv run --no-sync python -c "from sentence_transformers import CrossEncoder; CrossEncoder('$RERANKER_MODEL')"

COPY src ./src
COPY app.py ./
COPY .streamlit ./.streamlit
RUN uv sync --locked --no-dev


ENV HF_HUB_OFFLINE=1

CMD ["uv", "run", "--no-sync", "streamlit", "run", "app.py"]