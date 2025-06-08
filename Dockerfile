ARG PTH_DIR
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# Установка uv
RUN pip install uv

WORKDIR /inferd

# Установка зависимостей
COPY pyproject.toml .
RUN uv lock

# Копируем все исходники кроме моделей
COPY petals .

# Получаем имя нужного pth-файла через аргумент
ARG PTH_DIR

COPY model_parts/${PTH_DIR} model_parts/${PTH_DIR}

EXPOSE 6050 7050/udp 7051/udp

CMD ["uv", "run", "python", "run_node.py"]
