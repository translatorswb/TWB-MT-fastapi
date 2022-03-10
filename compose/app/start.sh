#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

uvicorn main:app --reload --reload-dir app --host 0.0.0.0 --port 8000
#--log-config logging.yml
