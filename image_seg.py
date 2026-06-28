import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# p2: preprocessing and normalization

image_bgr = cv2.imread('HW1_IMG_CS898BA.png') # load alien
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB) # convert to LAB color space
l_channel, a_channel, b_channel = cv2.split(image_lab) # split into L, A, B channels
l_eq = cv2.equalizeHist(l_channel) # equalize L channel
image_lab_eq= cv2.merge((l_eq, a_channel, b_channel)) # merge equalized L channel with A and B channels
image_normalized = cv2.cvtColor(image_lab_eq, cv2.COLOR_LAB2BGR) # convert back to BGR color space
cv2.imwrite('HW1_IMG_CS898BA_normalized.png', image_normalized) # save normalized image