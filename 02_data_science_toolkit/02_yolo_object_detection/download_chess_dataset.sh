#!/bin/bash

# Download pre-trained model along with sample images and config files
curl https://yolo-models.s3.us-east-2.amazonaws.com/chess_dataset.zip -o chess_dataset.zip
unzip chess_dataset.zip
rm chess_dataset.zip

# Move config and data files
cp chess_dataset/data/* yolov3/data 
cp chess_dataset/cfg/* yolov3/cfg 