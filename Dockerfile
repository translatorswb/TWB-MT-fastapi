FROM python:3.8-slim

# Project setup

ENV VIRTUAL_ENV=/opt/venv

WORKDIR /app

RUN apt-get update && apt-get clean

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install  --quiet --upgrade pip && \
    pip install  --quiet pip-tools
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app/

RUN python /app/nltk_pkg.py
