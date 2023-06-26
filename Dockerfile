FROM python:3.9

RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean;

WORKDIR /app

COPY . /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -m nltk.downloader punkt

CMD [ "python", "app.py" ]
