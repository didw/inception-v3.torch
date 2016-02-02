# Torch port of Inception V3

Scripts to dump TensorFlow [Inception V3](https://tensorflow.org/tutorials/image_recognition/) weights and to reconstruct the
network in Torch.

The approach is inspired by [soumith/inception.torch](https://github.com/soumith/inception.torch).

## Overview

* `dump_filters.py`: a Python/TensorFlow script to dump all the weights of Inception V3
* `inceptionv3.lua`: reads the weights and builds the Torch binary equivalent network
* `example.lua`: example use of the Torch network

## Usage

### Step 1: TensorFlow

Here are instructions using Docker:

```
# From the host
docker run -it \
-p 8890:8888 \
-v /home/didw/TF/code/inception-v3.torch/dump_filters.py:/root/dump_filters.py \
-v /home/didw/TF/data/dump:/root/dump \
b.gcr.io/tensorflow/tensorflow

# From the container
apt-get update
apt-get install -y libhdf5-dev
pip install h5py
python dump_filters.py
```

If you have already installed TensorFlow, just run `dump_filters.py` and the
script will generate a directory `dump` with all the filters.

### Step 2: Torch

Install pre-requisite:

```
luarocks install hdf5
```

Given that the filters are dumped in `/root/DeepLearning/torch/video_classification/caffemodel/dump`, execute:

```
luajit inceptionv3.lua -i /root/DeepLearning/torch/video_classification/caffemodel/dump \
-o /root/DeepLearning/torch/video_classification/caffemodel/networks/inceptionv3.net \
-b cunn
```

The parameter `-b` sets the backend to use: `nn`, `cunn`, or `cudnn`. The produced binary Torch model will
be saved in `/root/DeepLearning/torch/video_classification/caffemodel/networks/inceptionv3.net`.

Test it with an image as follows:

```
luajit example.lua -m /root/DeepLearning/torch/video_classification/caffemodel/networks/inceptionv3.net \
-b cudnn \
-i myimage.jpg \
-s synsets.txt
```

With TensorFlow [example image](https://www.tensorflow.org/versions/master/images/cropped_panda.jpg) you should obtain
a result like this:

```
RESULTS (top-5):
----------------
score = 0.847576: n02510455 giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (170)
score = 0.020494: n02500267 indri, indris, Indri indri, Indri brevicaudatus (76)
score = 0.003694: n02509815 lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (8)
score = 0.001323: n13044778 earthstar (879)
score = 0.001301: n07760859 custard apple (326)
```
