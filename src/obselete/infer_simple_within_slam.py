"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

print(" -- PYTHON: module imported successfully ...")

class MRCNN:

    def __init__(self):
        print(" -- PYTHON: object initialised successfully ...")

    def infer(self, im):
        if isinstance(im, str):
           print(" -- PYTHON: using char* ")
           img = mpimg.imread(im)
           imgplot = plt.imshow(img)
           plt.show()
        else:
           print(" -- PYTHON: using Mat ")
           #print(im)
           #print(im.shape)
           img = Image.fromarray(im, 'RGB')
           img.save('my.png')
           img.show()
