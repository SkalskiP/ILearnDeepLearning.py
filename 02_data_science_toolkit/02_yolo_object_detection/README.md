## Chess, rolls or basketball? Let's create a custom object detection model

My posts on the Medium are usually very theoretical - I tend to analyse and describe the algorithms that define how Neural Networks work. This time, however, I decided to break this trend and show my readers how easy it is to train your own YOLO model, capable of detecting any objects we choose. In order to achieve this goal, we will need help from a very useful and easy-to-use [implementation][1] of YOLO. In short, not much coding, but a huge effect.

### Setup environment

**LINUX** users can set up the environment with a single command. The script will download the YOLO implementation for you, and install all dependencies inside the Python virtual environment.

``` bash
# Run in 02_yolo_object_detection directory
bash setup_yolo_framework.sh
```

**Windows** and **MacOS** users should use Docker to build their environment.

``` bash
# Run in 02_yolo_object_detection directory
docker build -t yolo -f Dockerfile .
# Run docker image in interactive mode
docker run -it yolo:latest
```

### Detection using pre-trained chess model

``` bash
# Run in 02_yolo_object_detection directory
bash download_chess_model.sh
cd yolov3
# Run with activated virtual environment
python3 detect.py \
  --cfg cfg/chess.cfg \
  --weights ../chess/weights/best.pt \
  --source ../chess/samples \
  --names data/chess.names
```

<p align="center"> 
    <img width="700" src="./visualizations/object_detection_chess.gif" alt="Convolution">
</p>

<p align="center"> 
    <b>Figure 1.</b> Chess detection using TinyYOLO
</p>

### Detection using pre-trained rolls model

``` bash
# Run in 02_yolo_object_detection directory
bash download_rolls_model.sh
cd yolov3
# Run with activated virtual environment
python3 detect.py \
  --cfg cfg/rolls.cfg \
  --weights ../rolls/weights/best.pt \
  --source ../rolls/samples \
  --names data/rolls.names
```

<p align="center"> 
    <img width="700" src="./visualizations/object_detection_rolls.gif" alt="Convolution">
</p>

<p align="center"> 
    <b>Figure 2.</b> Roll's detection using TinyYOLO
</p>

### Detection using pre-trained basketball model

``` bash
# Run in 02_yolo_object_detection directory
bash download_basketball_model.sh
cd yolov3
# Run with activated virtual environment
python3 detect.py \
  --cfg cfg/basketball.cfg \
  --weights ../basketball/weights/best.pt \
  --source ../basketball/samples \
  --names data/basketball.names
```

<p align="center"> 
    <img width="700" src="./visualizations/object_detection_basketball.gif" alt="Convolution">
</p>

<p align="center"> 
    <b>Figure 3.</b> Detection of players moving around the basketball court, </br> based on <a href="https://research.google.com/youtube8m/">YouTube-8M</a> dataset.
</p>

### Download chess dataset and train model on your own

``` bash
# Run in 02_yolo_object_detection directory
bash download_chess_dataset.sh
cd yolov3
# Run with activated virtual environment
python3 train.py \
  --data data/chess.data \
  --cfg cfg/chess.cfg \
  --weights weights/yolo3.pt \
  --epochs 100
```

[1]: https://github.com/ultralytics/yolov3
