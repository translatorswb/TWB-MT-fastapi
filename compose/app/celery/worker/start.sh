#!/bin/bash

set -o errexit
set -o nounset

#celery -A main.celery worker \
#    --loglevel=info \
#    --max-tasks-per-child 1 \
#    --autoscale 1,2 \
#    --uid=${UWSGI_USER} \
#    --gid=${UWSGI_GROUP} \

python main.py
