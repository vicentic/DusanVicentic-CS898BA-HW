import cv2
import numpy as np
import scipy.stats as stats
import matplotlib as plt
import os

img = cv2.imread('/Users/dusanvicentic/WSU/Image Analysis & Computer Vision/DusanVicentic-CS898BA-Project1/HW1_IMG_CS898BA.png')

b, g, r = cv2.split(img)
channels = {'Blue': b, 'Green': g, 'Red': r}

# Descriptive statistics for each channel
for name, channel in channels.items():
    flat = channel.flatten()

    print(f"--- Stats for {name} Channel ---")
    print(f'Min: {np.min(flat)}')
    print(f'Max: {np.max(flat)}')
    print(f'Range: {np.max(flat)}')
    print(f'Average: {np.mean(flat):.2f}')
    print(f'Median: {np.median(flat)}')

    mode_result = stats.mode(flat, keepdims=False)
    print(f'Mode: {mode_result.mode}')
    print(f'Skew: {stats.skew(flat):.2f}')

    print(f'Std Dev: {np.std(flat):.2f}')
    print(f'Variance: {np.var(flat):.2f}\n')

# Conversions
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

base_images = [img, gray, binary, hsv, lab, hls]

# Histogram equalization on V(value)
h, s, v = cv2.split(hsv)
v_equalized = cv2.equalizeHist(v)
hsv_equal = cv2.merge((h, s, v_equalized))

# Convert image to RGB
final_eq_img = cv2.cvtColor(hsv_equal, cv2.COLOR_HSV2BGR)

# Added to list of images
base_images.append(final_eq_img)

