import sys
import os.path

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#
# # Load a color image
# img = cv2.imread('datasets/apple/apple_20.png')
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
# rgb = io.imread('datasets/apple/apple_20.png')
# lab = color.rgb2lab(rgb)
# print rgb

from io_util.image import loadRGB, loadLab,saveGray
import numpy as np

image = loadLab('datasets/apple/apple_55.png')
image = (1.0 / 255.0) * np.float32(image)
# print '---'
# from core.color_pixels import ColorPixels
# color_pixels = ColorPixels(loadRGB('datasets/apple/apple_20.png'))
# temp = color_pixels.pixels('Lab')
# print temp


img = np.zeros([image.shape[0],image.shape[1]],dtype=np.float32)
# for ii in  range(100):
#     for i in range(len(temp)):
#         img[ii][i] = temp[i]

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        img[i,j] = image[i,j,0]

from cv.image import rgb
img = rgb(img)
print img * 255



cv2.imshow('gray_image', img)
cv2.imwrite('test.png', img * 255)

cv2.waitKey(0)
