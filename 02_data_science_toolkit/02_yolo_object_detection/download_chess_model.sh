#!/bin/bash

# Download pre-trained model along with sample images and config files
curl https://yolo-models.s3.us-east-2.amazonaws.com/chess.zip -o chess.zip
unzip chess.zip
rm chess.zip

# Move config and data files
cp chess/data/* yolov3/data 
cp chess/cfg/* yolov3/cfg 