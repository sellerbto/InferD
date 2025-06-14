FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /inferd

COPY pyproject.toml .
RUN uv sync

COPY petals .

EXPOSE 6050 7050/udp 7051/udp

ENTRYPOINT ["uv", "run", "--no-sync", "python"]
