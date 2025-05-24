FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
RUN pip install uv

WORKDIR /inferd

COPY . .

RUN uv lock

EXPOSE 6050 7050/udp 7051/udp

CMD ["uv", "run", "python", "run_node.py"]