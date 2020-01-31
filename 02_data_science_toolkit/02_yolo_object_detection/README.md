## Chess, rolls or basketball? Let's create a custom object detection model

### Hit the ground running

``` bash
# Run in 02_yolo_object_detection directory
bash setup_yolo_framework.sh
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
    <img width="500" src="./visualizations/object_detection_chess.gif" alt="Convolution">
</p>

<p align="center"> 
    <b>Figure 2.</b> Chess detection using TinyYOLO
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
    <img width="500" src="./visualizations/object_detection_rolls.gif" alt="Convolution">
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
    <img width="500" src="./visualizations/object_detection_basketball.gif" alt="Convolution">
</p>

<p align="center"> 
    <b>Figure 3.</b> Detection of players moving around the basketball court, </br> based on <a href="https://research.google.com/youtube8m/">YouTube-8M</a> dataset.
</p>
