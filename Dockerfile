FROM python:3.8-slim

# Project setup

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get clean

RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

COPY . /app/

RUN python /app/nltk_pkg.py
