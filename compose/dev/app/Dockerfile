FROM python:3.8-slim

# Project setup

ENV VIRTUAL_ENV=/opt/venv \
    UWSGI_USER=twb \
    UWSGI_GROUP=twb

RUN apt-get update \
    && apt-get install -y telnet netcat \
    && apt-get clean

#################################
# Create local user UWSGI_USER` #
#################################
RUN adduser --disabled-password --gecos '' "$UWSGI_USER"

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --quiet --upgrade pip && \
    pip install --quiet pip-tools
COPY ./app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY ./compose/dev/app/start.sh /start.sh
RUN chmod +x /start.sh

COPY ./compose/dev/app/celery/worker/start.sh /start-celeryworker
RUN chmod +x /start-celeryworker

WORKDIR /app

COPY ./app/nltk_pkg.py /app/nltk_pkg.py
RUN python /app/nltk_pkg.py
