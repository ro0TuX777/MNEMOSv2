FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r /app/requirements.lock.txt
COPY server.py /app/server.py
COPY openapi.json /app/openapi.json

ENV MNEMOS_BASE_URL=http://host.docker.internal:8700
ENV MCP_TRANSPORT=stdio
ENV MCP_PORT=9700

CMD ["python", "/app/server.py"]
