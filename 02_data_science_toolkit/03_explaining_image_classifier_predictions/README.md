# Knowing What and Why? - Explaining Image Classifier Predictions

## Description

As we implement highly responsible Computer Vision systems, it is becoming progressively clear that we must provide not only predictions but also explanations, as to what influenced its decision. In this post, I compared and benchmarked the most commonly used libraries for explaining the model predictions in the field of Image Classification - [Eli5][1], [LIME][2], and [SHAP][3]. I investigated the algorithms that they leverage, as well as compared the efficiency and quality of the provided explanations.

## Hit the ground running

Via Conda
```sh
# setup conda environment & install all required packages
conda env create -f environment.yml
# activate conda environment
conda activate ExplainingImageClassifiers
```

Via Virtualenv
```sh
# set up python environment
apt-get install python3-venv
python3 -m venv .env
# activate python environment
source .env/bin/activate
# install all required packages
pip install -r requirements.txt
```

## Download COCO Dataset

```sh
cd 01_coco_res_net
sh get_coco_dataset_sample.sh
```

## ELI5 example

<p align="center"> 
    <img width="700" src="./01_coco_res_net/viz/coco_resnet34_eli5.png" alt="Eli5">
</p>

<p align="center"> 
    <b>Figure 1.</b> Explanations provided by ELI5
</p>

[1]: https://github.com/TeamHG-Memex/eli5
[2]: https://github.com/marcotcr/lime
[3]: https://github.com/slundberg/shap
