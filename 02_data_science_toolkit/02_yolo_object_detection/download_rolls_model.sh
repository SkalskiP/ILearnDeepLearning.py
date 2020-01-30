#!/bin/bash

# Download pre-trained model along with sample images and config files
curl https://yolo-models.s3.us-east-2.amazonaws.com/rolls.zip -o rolls.zip
unzip rolls.zip
rm rolls.zip

# Move config and data files
cp rolls/data/* yolov3/data 
cp rolls/cfg/* yolov3/cfg 