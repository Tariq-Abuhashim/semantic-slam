"""Performs testing of python-C++ bindings on a single image
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

print(" -- PYTHON: module imported successfully ...")

class MRCNN:

    def __init__(self): # no inputs, set to NULL
        print(" -- PYTHON: object initialised successfully ...")

    def display(self, im):
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
