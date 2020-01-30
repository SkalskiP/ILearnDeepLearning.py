#!/bin/bash

# Download pre-trained model along with sample images and config files
curl https://yolo-models.s3.us-east-2.amazonaws.com/basketball.zip -o basketball.zip
unzip basketball.zip
rm basketball.zip

# Move config and data files
cp basketball/data/* yolov3/data 
cp basketball/cfg/* yolov3/cfg 