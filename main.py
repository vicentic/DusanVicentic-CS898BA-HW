import cv2
import numpy as np
import scipy.stats as stats
import matplotlib as plt
import os

img_path = 'HW1_IMG_CS898BA.png'
img = cv2.imread(img_path)

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

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
base_names = ['original', 'gray', 'binary', 'hsv', 'lab', 'hls', 'equalized']

# Save base images
for image, name in zip(base_images, base_names):
    cv2.imwrite(f"{output_dir}/{name}.jpg", image)

# Random affine transformations 
import random
transformed_images = []
height, width = img. shape[:2]
center = (width / 2, height / 2)

for i, image in enumerate(base_images):
    for j in range(2):
        angle = random.randint(1, 359)
        scale = random.uniform(0.5, 1.5)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        warped = cv2.warpAffine(image, matrix, (width, height))
        transformed_images.append(warped)

total_images = base_images + transformed_images
# print(len(total_images))

#Applying Gaussian blur
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
blurred_images = []

for image in total_images:
    for s in sigmas:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=s)
        blurred_images.append(blur)

all_images = total_images + blurred_images
# 4 random subsets
subset = random.sample(all_images, 42)

# Greyscale for edge detection
for image in subset:
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image

# Edge detection
   # laplacian = cv2.Laplacian(gray_img
