#!/bin/bash

# Clone framework
git clone https://github.com/ultralytics/yolov3.git

# Getting pre-trained weights
cd ./yolov3/weights
sh download_yolov3_weights.sh
cd ..

# Setup virtual env
sudo apt-get update
sudo apt-get install python-dev
sudo apt-get install python-pip
sudo apt-get install python3-venv
python3 -m venv .env                            
source .env/bin/activate                      
pip3 install --upgrade pip
pip3 install --upgrade setuptools
pip3 install Cython
pip3 install numpy
pip3 install Pillow
pip3 install -r requirements.txt     
deactivate

echo "*****************************************************************************"
echo "******                Your environment has been created                ******"
echo "*****************************************************************************"
echo ""
echo "If you had no errors, You can proceed to work with your virtualenv as normal."
echo "(run 'source .env/bin/activate' in your assignment directory to load the venv,"
echo " and run 'deactivate' to exit the venv. See assignment handout for details.)"