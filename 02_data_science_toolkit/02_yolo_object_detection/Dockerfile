FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y apt-utils && \
    apt-get install -y git && \
    apt-get install -y unzip && \
    apt-get install -y curl && \
    apt-get install -y python-dev && \
    apt-get install -y python3-pip && \
    apt-get install -y libgtk2.0-dev

RUN mkdir project
COPY . project
WORKDIR project

RUN git clone https://github.com/ultralytics/yolov3.git
WORKDIR yolov3

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools && \
    pip3 install Cython && \
    pip3 install numpy && \
    pip3 install "pillow<7" && \
    pip3 install -r requirements.txt && \
    pip3 install torchvision

ENV DEBIAN_FRONTEND teletype