#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

gunicorn app.asgi:app -w 1 \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --chdir=/app
