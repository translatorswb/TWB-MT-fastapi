#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

gunicorn app.asgi:app -w ${GUNICORN_WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --chdir=/app \
    --user ${TWB_USER} \
    --group ${TWB_GROUP}
