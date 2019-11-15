FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD bear-classifier.pkl bear-classifier.pkl
ADD classifier.py classifier.py

EXPOSE 8008

CMD [ "python", "classifier.py" ]
