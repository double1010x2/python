#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : image_demo.py
#   Author      : YunYang1994
#   Created date: 2019-01-20 16:06:06
#   Description :
#
#================================================================

import cv2
import numpy as np
import core.utils as utils
import tensorflow.compat.v1 as tf
from PIL import Image
import sys
try:
    pb_file, image_path  = sys.argv[1], sys.argv[2]
except:
    print ("no image and model(*.pb) in argument!")
    raise RuntimeError
print ("model(*.pb): %s" %pb_file)
print ("image: %s" %image_path)
return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
#pb_file         = "./yolov3_coco.pb"
#image_path      = "/Users/vincentwu/Documents/GitHub/1st-DL-CVMarathon/homework/finalProject/data/kangaroo/images/00011.jpg" #./docs/images/road.jpeg"
num_classes     = 2 #80
input_size      = 416
graph           = tf.Graph()
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]
image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...]

return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)


with tf.Session(graph=graph) as sess:
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
                feed_dict={ return_tensors[0]: image_data})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
print (original_image_size)
print (input_size)
import pdb; pdb.set_trace()
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = utils.nms(bboxes, 0.35, method='nms')
image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()




