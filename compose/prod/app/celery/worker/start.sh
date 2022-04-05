#!/bin/bash

set -o errexit
set -o nounset

celery -A app.asgi.celery worker \
    --loglevel=info \
    --max-tasks-per-child 1 \
    --autoscale 1,2 \
    --uid=${TWB_USER} \
    --gid=${TWB_GROUP} \
