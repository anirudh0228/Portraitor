#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Segment images using weights from Fisher Yu (2016). Defaults to
settings for the Pascal VOC dataset.
'''

from __future__ import print_function, division

import skimage.morphology
import argparse
import os
import sys
import numpy as np
from PIL import Image,ImageFilter,ImageEnhance
from IPython import embed
from model import get_frontend, add_softmax, add_context

import matplotlib.pyplot as plt

#sys.path.append('C:/programming/data/models-master/research/object_detection')
#import object_detection_api as obj
# Settings for the Pascal dataset
input_width, input_height = 900, 900
label_margin = 186

has_context_module = False


def interp_map(prob, zoom, width, height):
    zoom_prob = np.zeros((height, width, prob.shape[2]), dtype=np.float32)
    for c in range(prob.shape[2]):
        for h in range(height):
            for w in range(width):
                r0 = h // zoom
                r1 = r0 + 1
                c0 = w // zoom
                c1 = c0 + 1
                rt = float(h) / zoom - r0
                ct = float(w) / zoom - c0
                v0 = rt * prob[r1, c0, c] + (1 - rt) * prob[r0, c0, c]
                v1 = rt * prob[r1, c1, c] + (1 - rt) * prob[r0, c1, c]
                zoom_prob[h, w, c] = (1 - ct) * v0 + ct * v1
    return zoom_prob

def get_trained_model(args):
    """ Returns a model with loaded weights. """
    
    model = get_frontend(input_width, input_height)

    if has_context_module:
        model = add_context(model)

    model = add_softmax(model)

    def load_tf_weights():
        """ Load pretrained weights converted from Caffe to TF. """

        # 'latin1' enables loading .npy files created with python2
        weights_data = np.load(args.weights_path, encoding='latin1').item()

        for layer in model.layers:
            if layer.name in weights_data.keys():
                layer_weights = weights_data[layer.name]
                layer.set_weights((layer_weights['weights'],
                                   layer_weights['biases']))

    def load_keras_weights():
        """ Load a Keras checkpoint. """
        model.load_weights(args.weights_path)

    if args.weights_path.endswith('.npy'):
        load_tf_weights()
    elif args.weights_path.endswith('.hdf5'):
        load_keras_weights()
    else:
        raise Exception("Unknown weights format.")

    return model



def forward_pass(args):
    ''' Runs a forward pass to segment the image. '''

    model = get_trained_model(args)
    img_original=Image.open(args.input_path)
    print(89)
    basewidth=300

              
    wpercent=(basewidth/float(img_original.size[0]))
    hsize=(float(img_original.size[1])*float(wpercent))
    img=img_original.resize((basewidth,int(hsize)),Image.ANTIALIAS)
    
    
#    object_extracted_img,object_extracted_img_large,xmin,ymin=obj.object_detection(img,img_original)
    
#    if(object_extracted_img_large.size[0]>object_extracted_img_large.size[1]):
#        basewidth=500
#        wpercent=(basewidth/float(object_extracted_img_large.size[0]))
#        hsize=(float(object_extracted_img_large.size[1])*float(wpercent))
#    else:
#        hsize=500
#        wpercent=(hsize/float(object_extracted_img_large.size[1]))
#        basewidth=(float(object_extracted_img_large.size[0])*float(wpercent))
   
        
#    object_extracted_img=object_extracted_img_large.resize((int(basewidth),int(hsize)),Image.ANTIALIAS)
#                                       
    # Load image and swap RGB -> BGR to match the trained weights
    image_rgb = np.array(img).astype(np.float32)
    image = image_rgb[:, :, ::-1] - args.mean                   
    image_size = image.shape
    print(103)
    # Network input shape (batch_size=1)
    net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)

    output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin

    # This simplified prediction code is correct only if the output
    # size is large enough to cover the input without tiling
    assert image_size[0] < output_height
    assert image_size[1] < output_width

    # Center pad the original image by label_margin.
    # This initial pad adds the context required for the prediction
    # according to the preprocessing during training.
    image = np.pad(image,
                   ((label_margin, label_margin),
                    (label_margin, label_margin),
                    (0, 0)), 'reflect')

    # Add the remaining margin to fill the network input width. This
    # time the image is aligned to the upper left corner though.
    margins_h = (0, input_height - image.shape[0])
    margins_w = (0, input_width - image.shape[1])
    image = np.pad(image,
                   (margins_h,
                    margins_w,
                    (0, 0)), 'reflect')

    # Run inference
    net_in[0] = image
    prob = model.predict(net_in)[0]
    print(135)
    # Reshape to 2d here since the networks outputs a flat array per channel
    prob_edge = np.sqrt(prob.shape[0]).astype(np.int)
    prob = prob.reshape((prob_edge, prob_edge, 21))

    # Upsample
    if args.zoom > 1:
        prob = interp_map(prob, args.zoom, image_size[1], image_size[0])

    # Recover the most likely prediction (actual segment class)
    prediction = np.argmax(prob, axis=2)
    print(146)  
  
    
    prediction=(prediction).astype(np.uint8)                          
    print(153)
                
    predicted_img=Image.fromarray(prediction,'L')


#    predicted_img=predicted_img.resize((object_extracted_img_large.size[0],object_extracted_img_large.size[1]),Image.ANTIALIAS)
    predicted_img=predicted_img.resize((img_original.size[0],img_original.size[1]),Image.ANTIALIAS)
#    original_crop=img_original.crop((xmin*img_original.size[0]/basewidth,ymin*img_original.size[1]/hsize,predicted_img.size[0]-xmin*img_original.size[0]/basewidth,predicted_img.size[1]-ymin*img_original.size[1]/hsize))
    prediction=np.array(predicted_img)
    prediction_mask=(prediction.squeeze()==15)
#    prediction_mask_img=Image.fromarray(prediction_mask.astype(np.uint8))
#    prediction_mask_img.save('C:/programming/codes/segmentation_keras-master/images/Anirudh2.jpg')
#    
    cropped_object=img_original*np.dstack((prediction_mask,)*3)
#    cropped_object=object_extracted_img_large*np.dstack((prediction_mask,)*3)
    
    img_original=img_original.filter(ImageFilter.GaussianBlur(9))

    img_original=img_original.convert("RGBA")
    img=img.filter(ImageFilter.BLUR)
    cropped_img=Image.fromarray(cropped_object)
    
    cropped_img.save('C:/programming/codes/segmentation_keras-master/images/Anirudh2.jpg')   
    cropped_img=cropped_img.convert("RGBA")
#    object_extracted_img_large=object_extracted_img_large.convert("RGBA")

    pix_data=cropped_img.load()
    wid,hei=cropped_img.size
    
    
    
    for y in range(hei):
        for x in range(wid):
            if pix_data[x,y]==(0,0,0,255):
                pix_data[x,y]=(0,0,0,0)
    
    print(199)

#    object_extracted_img_large=object_extracted_img_large.filter(ImageFilter.GaussianBlur(20))
#    img_original=Image.alpha_composite(object_extracted_img_large,cropped_img)
    img_original=Image.alpha_composite(img_original,cropped_img)
   
    print('Saving results to: ', args.output_path)
    with open(args.output_path, 'wb') as out_file:
        img_original.save(out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', nargs='?', default='C:/programming/codes/segmentation_keras-master/images/Anirudh1.jpg',
                        help='Required path to input image')
    parser.add_argument('--output_path', default=None,
                        help='Path to segmented image')
    parser.add_argument('--mean', nargs='*', default=[102.93, 111.36, 116.52],
                        help='Mean pixel value (BGR) for the dataset.\n'
                             'Default is the mean pixel of PASCAL dataset.')
    parser.add_argument('--zoom', default=8, type=int,
                        help='Upscaling factor')
    parser.add_argument('--weights_path', default='C:/programming/codes/segmentation_keras-master/conversion/converted/dilation8_pascal_voc.npy',
                        help='Weights file')

    args = parser.parse_args()
    if not args.output_path:
        dir_name, file_name = os.path.split(args.input_path)
        args.output_path = os.path.join(
            dir_name,
            '{}_seg.png'.format(
                os.path.splitext(file_name)[0]))

    forward_pass(args)


if __name__ == "__main__":
    main()
