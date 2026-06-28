import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# p2: preprocessing and normalization

image_bgr = cv2.imread('HW1_IMG_CS898BA.png') # load alien
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB) # convert to LAB color space
l_channel, a_channel, b_channel = cv2.split(image_lab) # split into L, A, B channels
l_eq = cv2.equalizeHist(l_channel) # equalize L channel
image_lab_eq = cv2.merge((l_eq, a_channel, b_channel)) # merge equalized L channel with A and B channels
image_norm = cv2.cvtColor(image_lab_eq, cv2.COLOR_LAB2BGR) # convert back to BGR color space
cv2.imwrite('HW2_IMG_CS898BA_normalized.png', image_norm) # save normalized image

# p3: Threshold segmentation

gray_norm = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY) # convert normalized image to grayscale
ret, mask_otsu = cv2.threshold(gray_norm, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # apply Otsu's thresholding

mask_adapt = cv2.adaptiveThreshold(
    gray_norm, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
) # adaptive thresholding (11x11 pixels, 2 const subtracted from mean)

cv2.imwrite('HW2_IMG_CS898BA_mask_otsu.png', mask_otsu)
cv2.imwrite('HW2_IMG_CS898BA_mask_adaptive.png', mask_adapt)

# p4: Color-space clustering (K-means)

image_hsv = cv2.cvtColor(image_norm, cv2.COLOR_BGR2HSV) # convert normalized image to HSV color space
pixel_values = image_hsv.reshape((-1, 3)) # reshape to a 2d array of pixels
pixel_values = np.float32(pixel_values) # convert to float

# k-means criteeria (k=4)
criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.2)
k = 3
''' Tuning k
for k in range(3, 6):
    print(f'K-means clustering with k={k}')
'''
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segment_labels = labels.reshape(image_hsv.shape[:2]) # reshape labels to original image

''' finding alien cluster
for i in range(k):
    mask = np.uint8(segment_labels == i) * 255
    cv2.imwrite(f'{k}_cluster_test_{i}.png', mask)
'''

alien_cluster_id = 0 # select alien mask
mask_kmeans = np.uint8(segment_labels == alien_cluster_id) * 255 # create mask for alien cluster
cv2.imwrite('HW2_IMG_CS898BA_mask_kmeans.png', mask_kmeans) # save k-means mask

