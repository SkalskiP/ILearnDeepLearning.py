<h1 align="center">ILearnDeepLearning.py</h1>

## Description

This repository contains small projects mainly related to Deep Learning but also Data Science in general. Subjects are closely linekd with articles I publish on [Medium][1] and are intended to complement the blog posts. I hope that the content of the repository will turn out to be interesting and, above all, useful. I encourage you both to read as well as to check how the code works in the action.

## Hit the ground running

``` bash
# clone repository
git clone https://github.com/SkalskiP/ILearnDeepLearning.py.git

# navigate to main directory
cd ILearnDeepLearning.py

# set up and activate python environment
python3 -m venv .env
source .env/bin/activate

# install all required packages
pip install -r requirements.txt
```

## Deep Dive into Math Behind Deep Networks

This project is mainly focused on visualizing quite complex issues related to gradient descent, activation functions and visualization of classification boundaries while teaching the model. It is a code that complements the issues described in more detail in the article [Deep Dive into Math Behind Deep Networks][2]. Here are some of the visualizations that have been created.

<p align="center"> 
    <img height="250" src="./01_mathematics_of_nn/final_visualisations/keras_frames.png" alt="Keras model frames">
    <img height="250" src="./01_mathematics_of_nn/final_visualisations/keras_class_boundaries.gif" alt="Keras class boundries">
</p>

<p align="center"> 
    <b>Figure 1.</b> A classification boundaries graph created in every iteration of the Keras model.</br>
    Finally, the frames were combined to create an animation.
</p>

<p align="center"> 
    <img width="400" src="./01_mathematics_of_nn/final_visualisations/gradient_descent.gif" alt="Gradient descent">
</p>

<p align="center"> 
    <b>Figure 2.</b> Visualization of the gradient descent.
</p>

## Preventing Deep Neural Network from Overfitting

This time I focused on the analysis of the reasons for overfitting and ways to prevent it. I made simulations of neural network regulation for different lambda coefficients, analyzing the change of values in the weight matrix. Take a look at the visualizations that were created in the process. Also check the article [Preventing Deep Neural Network from Overfitting][3].

<p align="center"> 
    <img width="500" src="./02_exploring_overfitting/final_visualisations/bias_variance.gif" alt="Change of accuracy">
</p>

<p align="center"> 
    <b>Figure 3.</b> Figure 3. Classification boundaries created by: top right corner - linear regression;</br>
    bottom left corner - neural network; bottom right corner - neural network with regularisation
</p>

<p align="center"> 
    <img width="500" src="./02_exploring_overfitting/final_visualisations/train_vs_validation_acc.gif" alt="Change of accuracy">
</p>

<p align="center"> 
    <b>Figure 4.</b> Change of accuracy values in subsequent epochs during neural network learning.
</p>

## Simple Method of Creating Animated Graphs

Both in my articles and projects I try to create interesting visualizations, which very often allow me to communicate my ideas much more effectively. I decided to create a short tutorial to show you how to easily create animated visualizations using Matplotlib. I also encourage you to read - [Simple Method of Creating Animated Graphs][4] - where I described, among other things, how to create a visualization of neural network learning process.

<p align="center"> 
    <img width="500" src="./03_animated_graphs/final_visualisations/lorenz_attractor.gif" alt="Change of accuracy">
</p>

<p align="center"> 
    <b>Figure 5.</b> Lorenz Attractor created using the Matplotlib animation API.
</p>

## License

This project is licensed under the MIT License - see the [LICENSE.md][6] file for details

## Interesting materials and ideas

This is a place where I collect links to interesting articles and papers, which I hope will become the basis for my next projects in the future.

1. [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings][5]
2. [Sequence to Sequence Learning with Neural Networks][7]
3. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation][8]

[1]: https://medium.com/@piotr.skalski92
[2]: https://towardsdatascience.com/https-medium-com-piotr-skalski92-deep-dive-into-deep-networks-math-17660bc376ba
[3]: https://towardsdatascience.com/preventing-deep-neural-network-from-overfitting-953458db800a
[4]: https://towardsdatascience.com/simple-method-of-creating-animated-graphs-127c11f58cc5
[5]: https://arxiv.org/abs/1607.06520
[6]: ./LICENSE.md
[7]: https://arxiv.org/abs/1409.3215
[8]: https://arxiv.org/abs/1406.1078
