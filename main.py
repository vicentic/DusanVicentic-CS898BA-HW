import cv2
import numpy as np
import scipy.stats as stats
import matplotlib as plt
import os

img = cv2.imread('/Users/dusanvicentic/WSU/Image Analysis & Computer Vision/DusanVicentic-CS898BA-Project1/HW1_IMG_CS898BA.png')

b, g, r = cv2.split(img)
channels = {'Blue': b, 'Green': g, 'Red': r}

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