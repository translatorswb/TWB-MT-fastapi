FROM python:3.8-slim

# Project setup

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update \
    && apt-get install -y telnet netcat \
    && apt-get clean

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --quiet --upgrade pip && \
    pip install --quiet pip-tools
COPY ./app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY ./compose/app/start.sh /start.sh
RUN chmod +x /start.sh

COPY ./compose/app/celery/worker/start.sh /start-celeryworker
RUN chmod +x /start-celeryworker

WORKDIR /app

COPY ./app/nltk_pkg.py /app/nltk_pkg.py
RUN python /app/nltk_pkg.py
