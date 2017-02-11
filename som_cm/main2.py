import sys
import os.path

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#
# # Load a color image
# img = cv2.imread('datasets/apple/apple_0.png')
#
#
#
#
# #Display the color image
# cv2.imshow('color_image',img)
#
#
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#
# #convert RGB image to Gray
# gray=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
#
# #Display the gray image
# cv2.imshow('gray_image',gray)
#
# from skimage import io, color
# rgb = io.imread('datasets/apple/apple_0.png')
# lab = color.rgb2lab(rgb)
# print rgb

from io_util.image import loadRGB, loadLab
import numpy as np

image = loadLab('datasets/apple/apple_0.png')
print image
print '---'
from core.color_pixels import ColorPixels
color_pixels = ColorPixels(loadRGB('datasets/apple/apple_0.png'))
temp = color_pixels.pixels('Lab')
print temp

img = np.zeros([100,len(temp),3],dtype=np.float32)
for ii in  range(100):
    for i in range(len(temp)):
        img[ii][i] = temp[i]

from cv.image import Lab2rgb
img = Lab2rgb(img)



cv2.imshow('gray_image', img)

cv2.waitKey(0)
