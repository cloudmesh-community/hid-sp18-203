FROM python:3-onbuild

RUN apt-get update
RUN apt-get install -q -y net-tools

COPY . /app
WORKDIR /app
EXPOSE 8080
CMD ["make", "run"]