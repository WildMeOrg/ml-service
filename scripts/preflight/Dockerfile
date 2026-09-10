ARG RUNTIME_IMAGE=ml-service
FROM ${RUNTIME_IMAGE}
COPY scripts/preflight/requirements.txt /tmp/preflight-requirements.txt
RUN python3 -m pip freeze > /tmp/runtime-constraints.txt && \
    python3 -m pip install --no-cache-dir -c /tmp/runtime-constraints.txt \
    -r /tmp/preflight-requirements.txt
COPY scripts/preflight/ /app/scripts/preflight/
