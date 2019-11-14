FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
ADD classifier.py classifier.py
ADD bear-classifier.pkl bear-classifier.pkl

RUN pip install -r requirements.txt

EXPOSE 8008

RUN python classifier.py
