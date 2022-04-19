FROM python:3.8-slim

# Project setup

ENV VIRTUAL_ENV=/opt/venv \
    TWB_USER=twb \
    TWB_GROUP=twb

# Create local user TWB_USER
RUN adduser --disabled-password --gecos '' "$TWB_USER"

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --quiet --upgrade pip && \
    pip install --quiet pip-tools
COPY ./app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY ./compose/prod/app/start.sh /start.sh
RUN chmod +x /start.sh

COPY ./compose/prod/app/celery/worker/start.sh /start-celeryworker
RUN chmod +x /start-celeryworker

COPY . /app
WORKDIR /app

COPY ./app/nltk_pkg.py /app/nltk_pkg.py
RUN python /app/nltk_pkg.py
