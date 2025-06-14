FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /inferd

COPY pyproject.toml .
RUN uv sync

COPY petals .

# ARG PTH_DIR

# COPY model_parts/${PTH_DIR} model_parts/${PTH_DIR}

EXPOSE 6050 7050/udp 7051/udp

ENTRYPOINT ["uv", "run", "--no-sync", "python"]
