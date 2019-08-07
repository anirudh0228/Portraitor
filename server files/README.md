Keras implementation of DilatedNet for semantic segmentation
============================================================

<div style="text-align: center" />
<img src="http://nicolovaligi.com/cat.jpg" style="max-width: 500px" />
</div>


A native Keras implementation of semantic segmentation according to
*Multi-Scale Context Aggregation by Dilated Convolutions (2016)*. Optionally uses the pretrained weights by the
[authors'](https://github.com/fyu/dilation).

The code has been tested on Tensorflow 1.3, Keras 1.2, and Python 3.6.


Using the pretrained model
----------------

Download and extract the pretrained model:

    curl -L https://github.com/nicolov/segmentation_keras/releases/download/model/nicolov_segmentation_model.tar.gz | tar xvf -

Install dependencies and run:

```
pip install -r requirements.txt
# For GPU support
pip install tensorflow-gpu==1.3.0

python predict.py --weights_path conversion/converted/dilation8_pascal_voc.npy
```

The output image will be under `images/cat_seg.png`.


Converting the original Caffe model
-----------------------------------

Follow the instructions in the `conversion` folder to convert the weights to the TensorFlow
format that can be used by Keras.


Training
--------

Download the *Augmented Pascal VOC* dataset
[here](http://home.bharathh.info/pubs/codes/SBD/download.html):

    curl -L http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz | tar -xvf -

This will create a `benchmark_RELEASE` directory in the root of the repo.
Use the `convert_masks.py` script to convert the provided masks in *.mat* format to RGB pngs:

    python convert_masks.py \
        --in-dir benchmark_RELEASE/dataset/cls \
        --out-dir benchmark_RELEASE/dataset/pngs

Start training:

    python train.py --batch-size 2

Model checkpoints are saved under `trained/`, and can be used with the `predict.py` script for testing.

The training code is currently limited to the *frontend* module,
and thus only outputs 16x16 segmentation maps. The augmentation
pipeline does mirroring but not cropping or rotation.

<hr>

*Fisher Yu and Vladlen Koltun, Multi-Scale Context Aggregation by Dilated Convolutions, 2016*
