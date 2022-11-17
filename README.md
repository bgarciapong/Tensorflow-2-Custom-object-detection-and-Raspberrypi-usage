# Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage
### Learn how to Train a TensorFlow Custom Object Detector with TensorFlow-GPU

This repository is a guide to use TensorFlow Object Detection API for training a custom object detector with TensorFlow 2 versions. ***As of 11/16/2022 I have tested with TensorFlow 2.8.0 to train a model on Windows 10 with a Nvidia 3080 Graphics Card.***

## Table of Content
1. [Installing Tensorflow GPU](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#installing-tensorflow-gpu)
2. [Workspace and Anaconda virtual enviroment](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#workspace-and-anaconda-virtual-enviroment)
3. [training Data](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#training-data)
4. [Training Pipeline](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#training-pipeline)
5. [Training model](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#training-model)
6. [Test Finished Model](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#test-finished-model)
7. [exporting the model](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#exporting-the-model)
8. [installing Tensorflow Nighly](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#installing-tensorflow-nighly)
9. [converting model to tensorflow Lite](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#converting-model-to-tensorflow-lite)
10. [Preparing our Model for Use](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/edit/main/README.md#preparing-our-model-for-use)

for this project I have used my own dataset which is a Card deck model.

### Installing Tensorflow GPU
- first step into installing what you need is to fisrt install anaconda by going to the following [link](https://www.anaconda.com/products/distribution) 
- you will now have to download CUDA and cuDNN these are tools that will utilize the graphics memory of the GPU and shift the workload. I recomend watching a video on how to donload these two. ***I downloaded CUDA version 11.5 and cuDNN version 8.3, this version worked with Tensorflow 2 version 2.8.0.

<p align="center">
  <img src="doc/pills.png">
</p>


### Workspace and Anaconda virtual enviroment

### training Data

### Training Pipeline

### Training model

### Test Finished Model

# Converting Tensorflow Models To Tensorflow Lite

### exporting the model

### installing Tensorflow Nighly

### converting model to tensorflow Lite

### Preparing our Model for Use
