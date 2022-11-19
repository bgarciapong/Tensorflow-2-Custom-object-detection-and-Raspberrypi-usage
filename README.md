# Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage
### Learn how to Train a TensorFlow Custom Object Detector with TensorFlow-GPU

This repository is a guide to use TensorFlow Object Detection API for training a custom object detector with TensorFlow 2 versions. ***As of 11/16/2022 I have tested with TensorFlow 2.8.0 to train a model on Windows 10 with a Nvidia 3080 Graphics Card.***

## Table of Content
1. [Installing Tensorflow GPU](https://github.com/bgarciapong/Tensorflow-2-Custom-object-detection-and-Raspberrypi-usage/blob/Master/README.md#installing-tensorflow-gpu)
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
  <img src="img/anaconda.png">
</p>


we will now create a virtual enviroment with this command

```
conda create -n tensorflow pip python=3.8
```
Then activate the environment with

```
conda activate tensorflow
```
**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

Once done with this we have everything needed to install TensorFlow-GPU (or TensorFlow CPU). So we can navigate back to our anaconda prompt, and issue the following command

```
pip install tensorflow-gpu
```

If you are installing TensorFlow CPU, instead use

```
pip install tensorflow
```
Once we are done with the installation, we can use the following code to check if everything installed properly

```
python
>>> import tensorflow as tf
>>> print(tf.__version__)
```
If everything has installed properly you should get the message, "2.8.0", or whatever version of TensorFlow you have. This means TensorFlow is up and running and we are ready to setup our workspace. We can now proceed to the next step!
**Note if there is an error with importing, you must install [Visual Studio 2019 with C++ Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16).**

### Workspace and Anaconda virtual enviroment
For the TensorFlow Object Detection API, there is a certain directory structure that we must follow to train our model. To make the process a bit easier, I added most of the necessary files in this repository.

create a folder directly in C: and name it "TesorFlow". 

```
cd C:\TensorFlow
```
Once you are here, you will have to clone the [TensorFlow models repository](https://github.com/tensorflow/models) with

```
git clone https://github.com/tensorflow/models.git
```

Download all files in the directory called models in the folder C:\Tensorflow and download [models file](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector/archive/master.zip). extract the files. 

Then, your directory structure should look something like this

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
└─ scripts/
└─ workspace/
   ├─ training_demo/
```

after you setup the structure, install the prerequisites for th object detection API. first to install the protofub compiler with 
```
conda install -c anaconda protobuf
```
Then you should cd in to the TensorFlow\models\research directory with

```
cd models\research
```
Then compile the protos with

```
protoc object_detection\protos\*.proto --python_out=.
```
after, you can close the terminal and open a new anaconda prompt. the activate the virtual environment again using.

```
conda activate tensorflow
```
With TensorFlow 2, pycocotools is a dependency for the Object Detection API. To install it with Windows Support use
```
pip install cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
**Note that Visual C++ 2015 build tools must be installed and on your path, according to the installation instructions. If you do not have this package, then download it [here](https://go.microsoft.com/fwlink/?LinkId=691126).**

Go back to the models\research directory with 

```
cd C:\TensorFlow\models\research
```

Once here, copy and run the setup script with 

```
copy object_detection\packages\tf2\setup.py .
python -m pip install .
```
If there are any errors, report an issue, but they are most likely pycocotools issues meaning your installation was incorrect. But if everything went according to plan you can test your installation with

```
python object_detection\builders\model_builder_tf2_test.py
```
You should get a similar output to this

```
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 45.304s

OK (skipped=1)
```
This means we successfully set up the Anaconda Directory Structure and TensorFlow Object Detection API. We can now finally collect and label our dataset. So, let's go on to the next step!


### training Data
now Tensorflow API is ready to go, we have to collect and labe picture that the model will be trained and tested on. All files that will be needed from now on are located on the workspace\training_demo directory.

- ```annotations```: This is where we will store all our training data needed for our model. By this I mean the CSV and RECORD files needed for the training pipeline. There is also a PBTXT File with the labels for our model. If you are training your own dataset you can delete train.record and test.record, but if you are training my Pill Classifier model you can keep them.
- ```exported-models```: This is our output folder where we will export and store our finished inference graph.
- ```images```: This folder consists of a test and train folder. Here we will store the labelled images needed for training and testing as you can probably infer. The labelled images consist of the original image and an XML File. If you want to train the Pill Classifier model, you can keep the images and XML documents, otherwise delete the images and XML files.
- ```models```: In this folder we will store our training pipeline and checkpoint information from the training job as well as the CONFIG file needed for training.
- ```pre-trained-models```: Here we will store our pre-trained model that we will use as a starting checkpoint for training
- The rest of the scripts are just used for training and exporting the model, as well as a sample object detection scipt that performs inference on a test image.

If you want to train a model on your own custom dataset, you must first gather images. Ideally you would want to use 100 images for each class. Say for example, you are training a cat and dog detector. if youre trainning your own model make sure to gather enoght pictures with different backgrounds, shading and angles.

After gathering some images, you must partition the dataset. By this I mean you must seperate the data in to a training set and testing set. You should put 80% of your images in to the images\training folder and put the remaining 20% in the images\test folder. After seperating your images, you can label them with [LabelImg](https://tzutalin.github.io/labelImg).

We have now gathered our dataset. This means we are ready to generate training data. So onwards to the next step!
we are ready to create the label_map. It is located in the annotations folder, so navigate to that within File Explorer. After you've located label_map.pbtxt, open it with a Text Editor of your choice. If you want to make your own custom object detector you must create a similar item for each of your labels. Since my model had two classes of pills, my labelmap looked like 

Example:
```
item {
    id: 1
    name: 'Acetaminophen 325 MG Oral Tablet'
}

item {
    id: 2
    name: 'Ibuprofen 200 MG Oral Tablet'
}
```
For example, if you wanted to make a basketball, football, and baseball detector, your labelmap would look something like
```
item {
    id: 1
    name: 'basketball'
}

item {
    id: 2
    name: 'football'
}

item {
    id: 3
    name: 'baseball'
}
```
Once you are done with this save as ```label_map.pbtxt``` and exit the text editor. Now we have to generate RECORD files for training. The script to do so is located in C:\TensorFlow\scripts\preprocessing, but we must first install the pandas package with
```
pip install pandas
```
Now we should navigate to the scripts\preprocessing directory with

```
cd C:\TensorFlow\scripts\preprocessing
```
Once you are in the correct directory, run these two commands to generate the records

```
python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\train -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\train.record

python generate_tfrecord.py -x C:\Tensorflow\workspace\training_demo\images\test -l C:\Tensorflow\workspace\training_demo\annotations\label_map.pbtxt -o C:\Tensorflow\workspace\training_demo\annotations\test.record
```
 After each command you should get a success message stating that the TFRecord File has been created. So now under ```annotations``` there should be a ```test.record``` and ```train.record```. That means we have generated all the data necessary, and we can proceed to configure the training pipeline in the next step
 
### Training Pipeline
For this tutorial, we will use a CONFIG File from one of the TensorFlow pre-trained models. There are plenty of models in the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), but we will use the [SSD MobileNet V2 FPNLite 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz), as it is on the faster end of the spectrum with decent performance. If you want you can choose a different model, but you will have to alter the steps slightly.

To download the model you want, just click on the name in the TensorFlow Model Zoo. This should download a tar.gz file. Once it has downloaded, extracts the contents of the file to the ```pre-trained-models``` directory. The structure of that directory should now look something like this

```
training_demo/
├─ ...
├─ pre-trained-models/
│  └─ ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```
Now, we must create a directory to store our training pipeline. Navigate to the ```models``` directory and create a folder called ```my_ssd_mobilenet_v2_fpnlite```. Then copy the ```pipeline.config``` from the pre-trained-model we downloaded earlier to our newly created directory. Your directory should now look something like this

```
training_demo/
├─ ...
├─ models/
│  └─ my_ssd_mobilenet_v2_fpnlite/
│     └─ pipeline.config
└─ ...
```

Then open up ```models\my_ssd_mobilenet_v2_fpnlite\pipeline.config``` in a text editor because we need to make some changes.
- Line 3. Change ```num_classes``` to the number of classes your model detects. For the basketball, baseball, and football, example you would change it to ```num_classes: 3```
- Line 135. Change ```batch_size``` according to available memory (Higher values require more memory and vice-versa). I changed it to:
  - ```batch_size: 6```
- Line 165. Change ```fine_tune_checkpoint``` to:
  - ```fine_tune_checkpoint: "pre-trained-models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0"```
- Line 171. Change ```fine_tune_checkpoint_type``` to:
  - ```fine_tune_checkpoint_type: "detection"```
- Line 175. Change ```label_map_path``` to:
  - ```label_map_path: "annotations/label_map.pbtxt"```
- Line 177. Change ```input_path``` to:
  - ```input_path: "annotations/train.record"```
- Line 185. Change ```label_map_path``` to:
  - ```label_map_path: "annotations/label_map.pbtxt"```
- Line 189. Change ```input_path``` to:
  - ```input_path: "annotations/test.record"```

Once we have made all the necessary changes, that means we are ready for training. So let's move on to the next step!
### Training model

Now you go back to your Anaconda Prompt. ```cd``` in to the ```training_demo``` with 

```
cd C:\TensorFlow\workspace\training_demo
```

I have already moved the training script in to the directory, so to run it just use 

```
python model_main_tf2.py --model_dir=models\my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models\my_ssd_mobilenet_v2_fpnlite\pipeline.config
```

When running the script, you should expect a few warnings but as long as they're not errors you can ignore them. Eventually when the training process starts you should see output similar to this

```
INFO:tensorflow:Step 100 per-step time 0.640s loss=0.454
I0810 11:56:12.520163 11172 model_lib_v2.py:644] Step 100 per-step time 0.640s loss=0.454
```

Awesome! You have officially started training your model! Now you can kick back and relax as this will take a few hours depending on your system. TensorFlow logs output similar to the one above every 100 steps of the process so if it looks frozen, don't worry about it. This output shows you two statistics: per-step time and loss. You're going to want to pay attention to the loss. In between logs, the loss tends to decrease. Your ideally going to want to stop the program when it's between 0.150 and 0.200. This prevents underfitting and overfitting. For me it took around 4000 steps before the loss entered that range. And then to stop the program just use CTRL+C.
### Exporting the Inference Graph

Once you have finished training and stopped the script, you are ready to export your finished model! You should still be in the ```training_demo``` directory but if not use

```
cd C:\TensorFlow\workspace\training_demo
```

I have already moved the script needed to export, so all you need to do is run this command

```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir .\models\my_ssd_mobilenet_v2_fpnlite\ --output_directory .\exported-models\my_mobilenet_model
```

**Note that if you get an error similar to ```TypeError: Expected Operation, Variable, or Tensor, got block4 in exporter_main_v2.py``` look at [this](https://github.com/tensorflow/models/issues/8881) error topic**

But if this program finishes successfully, then congratulations because your model is finished! It should be located in the ```C:\TensorFlow\workspace\training_demo\exported-models\my_mobilenet_model\saved_model``` folder. There should be an PB File called ```saved_model.pb```. This is the inference graph! I also prefer to copy the ```label_map.pbtxt``` file in to this directory because it makes things a bit easier for testing. If you forgot where the labelmap is located it should be in ```C:\TensorFlow\workspace\training_demo\annotations\label_map.pbtxt```. Since the labelmap and inference graph are organized, we are ready to test! 

### Test Finished Model
To test out your model, you can use the sample object detection script I provided called ```TF-image-od.py```. This should be located in ```C:\TensorFlow\workspace\training_demo```. **Update**: I have added video support, argument support, and an extra OpenCV method. The description for each program shall be listed below 
- ```TF-image-od.py```: This program uses the viz_utils module to visualize labels and bounding boxes. It performs object detection on a single image, and displays it with a cv2 window.
- ```TF-image-object-counting.py```: This program also performs inference on a single image. I have added my own labelling method with OpenCV which I prefer. It also counts the number of detections and displays it in the top left corner. The final image is, again, displayed with a cv2 window.
- ```TF-video-od.py```: This program is similar to the ```TF-image-od.py```. However, it performs inference on each individual frame of a video and displays it via cv2 window.
- ```TF-video-object-counting.py```: This program is similar to ```TF-image-object-counting.py``` and has a similar labelling method with OpenCV. Takes a video for input, and also performs object detection on each frame, displaying the detection count in the top left corner.

The usage of each program looks like 

```
usage: TF-image-od.py [-h] [--model MODEL] [--labels LABELS] [--image IMAGE] [--threshold THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Folder that the Saved Model is Located In
  --labels LABELS       Where the Labelmap is Located
  --image IMAGE         Name of the single image to perform detection on
  --threshold THRESHOLD Minimum confidence threshold for displaying detected objects
```
If the model or labelmap is located anywhere other than where I put them, you can specify the location with those arguments. You must also provide an image/video to perform inference on. If you are using my Pill Detection Model, this is unecessary as the default value should be fine. If you are using one of the video scripts, use ```--video``` instead of ```--image``` and provide the path to your test video. For example, the following steps run the sample ```TF-image-od.py``` script.

```
cd C:\TensorFlow\workspace\training_demo
```

Then to run the script, just use

```
python TF-image-od.py
``` 

**Note that if you get an error similar to ```
cv2.error: OpenCV(4.3.0) C:\Users\appveyor\AppData\Local\Temp\1\pip-req-build-kv3taq41\opencv\modules\highgui\src\window.cpp:651: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support. If you are on Ubuntu or Debian, install libgtk2.0-dev and pkg-config, then re-run cmake or configure script in function 'cvShowImage'
``` just run ```pip install opencv-python``` and run the program again**

If everything works properly you should get an output similar to this
<p align="center">
  <img src="doc/output.png">
</p>

There is also a webcam file in the C:\TensorFlow\workspace\training_demo directory, it allow you to activly test your model using a webcam. if you don't wish to use your trained model on a Raspberry pi you don't need to follow the next steps you have officially trained a object detection model that you should be able to use in your computer.

# Converting Tensorflow Models To Tensorflow Lite
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
### This Guide Contains Everything you Need to Convert your previously Custom and Pre-trained TensorFlow Models to TensorFlow Lite, 

**The following steps for conversion are based off of the directory structure and procedures in this guide. So if you haven't already taken a look at it, I recommend you do so.
To move on, you should have already**
  - **Installed Anaconda**
  - **Setup the Directory Structure**
  - **Compiled Protos and Setup the TensorFlow Object Detection API**
  - **Gathered Training Data**
  - **Trained your Model (without exporting)**
  

### exporting the model

If you haven't already, make sure you have already configured the training pipeline and trained the model. You should now have a training directory and a  Open up a new Anaconda terminal and activate the virtual environment we made in the other tutorial with

```
conda activate tensorflow
```
Now, we can change directories with

```
cd C:\TensorFlow\workspace\training_demo
```
Now, unlike my other guide, we aren't using ```exporter_main_v2.py``` to export the model. For TensorFlow Lite Models, we have to use ```export_tflite_graph_tf2.py```. You can export the model with
```
python export_tflite_graph_tf2.py --pipeline_config_path models\my_ssd_mobilenet_v2_fpnlite\pipeline.config --trained_checkpoint_dir models\my_ssd_mobilenet_v2_fpnlite --output_directory exported-models\my_tflite_model
```
**Note: At the moment, TensorFlow Lite only support models with the SSD Architecture (excluding EfficientDet). Make sure that you have trained with an SSD training pipeline before you continue. You can take a look at the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) or the [documentation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tf2.md) for the most up-to-date information.**

### installing Tensorflow Nighly

To avoid version conflicts, we'll first create a new Anaconda virtual environment to hold all the packages necessary for conversion. First, we must deactivate our current environment with

```
conda deactivate
```

Now issue this command to create a new environment for TFLite conversion.

```
conda create -n tflite pip python=3.7
```

We can now activate our environment with

```
conda activate tflite
```

**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

Now we must install TensorFlow in this virtual environment. However, in this environment we will not just be installing standard TensorFlow. We are going to install tf-nightly. This package is a nightly updated build of TensorFlow. This means it contains the very latest features that TensorFlow has to offer. There is a CPU and GPU version, but if you are only using it conversion I'd stick to the CPU version because it doesn't really matter. We can install it by issuing

```
pip install tf-nightly
```
Now, to test our installation let's use a Python terminal.
```
python
```
Then import the module with
```
(tflite) C:\Users\Brayan>python
Python 3.7.13 (default, Mar 28 2022, 08:03:21) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version__)
```

**Note: You might get an error with importing the newest version of Numpy. It looks something like this ```RuntimeError: The current Numpy installation ('D:\\Apps\\anaconda3\\envs\\tflite\\lib\\site-packages\\numpy\\__init__.py') fails to pass a sanity check due to a bug in the windows runtime. See this issue for more information: https://tinyurl.com/y3dm3h86```. You can fix this error by installing a previous version of Numpy with ```pip install numpy==1.19.3```.**

If the installation was successful, you should get the version of tf-nightly that you installed. 
```
2.11.0-dev20220812
```

### converting model to tensorflow Lite

Now, you might have a question or two. If the program is called ```export_tflite_graph_tf2.py```, why is the exported inference graph a ```saved_model.pb``` file? Isn't this the same as standard TensorFlow?
<p align="left">
  <img src="doc/saved_model.png">
</p>

Well, in this step we'll be converting the ```saved_model``` to a single ```model.tflite``` file for object detection with tf-nightly. I recently added a sample converter program to my other repository called ```convert-to-tflite.py```. This script takes a saved_model folder for input and then converts the model to the .tflite format. Additionally, it also quantizes the model. If you take a look at the code, there are also various different features and options commented. These are optional and might be a little buggy. For some more information, take a look at the [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert/). The usage of this program is as so

```
usage: convert-to-tflite.py [-h] [--model MODEL] [--output OUTPUT]

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    Folder that the saved model is located in
  --output OUTPUT  Folder that the tflite model will be written to
```

At the moment I'd recommend not using the output argument and sticking to the default values as it still has a few errors. Enough talking, to convert the model run
```
python convert-to-tflite.py
```

You should now see a file in the ```exported-models\my_tflite_model\saved_model``` directory called ```model.tflite```

<p align="left">
  <img src="doc/model.tflite.png">
</p>

Now, there is something very important to note with this file. Take a look at the file size of the ```model.tflite``` file. **If your file size is 1 KB, that means something has gone wrong with conversion**. If you were to run object detection with this model, you will get various errors. As you can see in the image, my model is 3,549 KB which is an appropriate size. If your file is significantly bigger, 121,000 KB for example, it will drastically impact performance while running. With a model that big, my framerates dropped all the way down to 0.07 FPS. If you have any questions about this, feel free to raise an issue and I will try my best to help you out. 

### Preparing our Model for Use
Now that we have our model, it's time to create a new labelmap. Unlike standard TensorFlow, TensorFlow uses a .txt file instead of a .pbtxt file. Creating a new labelmap is actually much easier than it sounds. Let's take a look at an example. Below, I have provided the ```label_map.pbtxt``` that I used for my Pill Detection model.
```
item {
    id: 1
    name: 'Acetaminophen 325 MG Oral Tablet'
}

item {
    id: 2
    name: 'Ibuprofen 200 MG Oral Tablet'
}
```
If we were to create a new labelmap for TensorFlow Lite, all we have to do is write each of the item names on it's own line like so
```
Acetaminophen 325 MG Oral Tablet
Ibuprofen 200 MG Oral Tablet
```
Once you are finished filling it out save the file within the ```exported-models\my_tflite_model\saved_model``` as ```labels.txt```. The directory should now look like this

<p align="left">
  <img src="doc/final model.png">
</p>
