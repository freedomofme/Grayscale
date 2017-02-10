import sys
import os.path

import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Load a color image
img = cv2.imread('datasets/apple/apple_0.png')




#Display the color image
cv2.imshow('color_image',img)


img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


#convert RGB image to Gray
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#Display the gray image
cv2.imshow('gray_image',gray)

cv2.waitKey(0)
