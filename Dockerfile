FROM python:3.8-slim

# Project setup

ENV VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get clean

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install  --quiet --upgrade pip && \
    pip install  --quiet pip-tools
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache/pip

#Custom translator requirements
#COPY ./app/customtranslators/<customtranslatorname>/requirements.txt /app/customrequirements.txt
#RUN pip install -r /app/customrequirements.txt \
#    && rm -rf /root/.cache/pip

COPY . /app
WORKDIR /app

COPY ./app/nltk_pkg.py /app/nltk_pkg.py
RUN python /app/nltk_pkg.py
