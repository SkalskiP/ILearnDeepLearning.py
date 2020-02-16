#!/bin/bash

# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh

mkdir dataset
cd dataset

mkdir coco
cd coco

mkdir images
cd images

# Download Images
wget -c https://pjreddie.com/media/files/val2014.zip

# Unzip
unzip -q val2014.zip

cd ..
cd ..

wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz

# Set Up Image Lists
paste <(awk "{print \"$PWD/coco\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
