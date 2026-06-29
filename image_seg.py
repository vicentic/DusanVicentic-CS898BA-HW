import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# p2: preprocessing and normalization

image_bgr = cv2.imread('HW1_IMG_CS898BA.png') # load alien
image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB) # convert to LAB color space
l_channel, a_channel, b_channel = cv2.split(image_lab) # split into L, A, B channels
l_eq = cv2.equalizeHist(l_channel) # equalize L channel
image_lab_eq = cv2.merge((l_eq, a_channel, b_channel)) # merge equalized L channel with A and B channels
image_norm = cv2.cvtColor(image_lab_eq, cv2.COLOR_LAB2BGR) # convert back to BGR color space
cv2.imwrite(os.path.join(output_dir, 'normalized.png'), image_norm) # save normalized image

# p3: Threshold segmentation

gray_norm = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY) # convert normalized image to grayscale
ret, mask_otsu = cv2.threshold(gray_norm, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # apply Otsu's thresholding

mask_adapt = cv2.adaptiveThreshold(
    gray_norm, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
) # adaptive thresholding (11x11 pixels, 2 const subtracted from mean)

cv2.imwrite(os.path.join(output_dir, 'mask_otsu.png'), mask_otsu)
cv2.imwrite(os.path.join(output_dir, 'mask_adaptive.png'), mask_adapt)

# p4: Color-space clustering (K-means)

image_hsv = cv2.cvtColor(image_norm, cv2.COLOR_BGR2HSV) # convert normalized image to HSV color space
pixel_values = image_hsv.reshape((-1, 3)) # reshape to a 2d array of pixels
pixel_values = np.float32(pixel_values) # convert to float

# k-means criteria (k=4)
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
cv2.imwrite(os.path.join(output_dir, 'mask_kmeans.png'), mask_kmeans) # save k-means mask

# p5 : plots

# load gt mask and convert to binary
gt_bin = cv2.imread('ground_truth.png', 0) > 127
otsu_bin = mask_otsu > 127
adaptive_bin = mask_adapt > 127
kmeans_bin = mask_kmeans > 127

# evaluations (IoU, Dice)
def calculate_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    iou = intersection / union if np.sum(union) != 0 else 0
    dice = (2.0 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) != 0 else 0
    return iou, dice

# Calculate the metrics
print('Evaluation Metrics:')

iou_otsu, dice_otsu = calculate_metrics(otsu_bin, gt_bin)
print(f'Otsu Thresholding - IoU: {iou_otsu:.4f}, Dice: {dice_otsu:.4f}')

iou_adapt, dice_adapt = calculate_metrics(adaptive_bin, gt_bin)
print(f'Adaptive Thresholding - IoU: {iou_adapt:.4f}, Dice: {dice_adapt:.4f}')

iou_kmeans, dice_kmeans = calculate_metrics(kmeans_bin, gt_bin)
print(f'K-means Clustering - IoU: {iou_kmeans:.4f}, Dice: {dice_kmeans:.4f}')

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(image_norm, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Normalized Image')
axes[0, 1].axis('off')

axes[1, 1].imshow(mask_otsu, cmap='gray')
axes[1, 1].set_title('Otsu Thresholding')
axes[1, 1].axis('off')

axes[1, 0].imshow(mask_adapt, cmap='gray')
axes[1, 0].set_title('Adaptive Thresholding')
axes[1, 0].axis('off')

axes[1, 2].imshow(mask_kmeans, cmap='gray')
axes[1, 2].set_title('K-means Clustering')
axes[1, 2].axis('off')

axes[0, 2].imshow(gt_bin, cmap='gray')
axes[0, 2].set_title('Ground Truth Mask')
axes[0, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hw2_segmentation_results.png'), dpi=300)
print('Segmentation results saved as hw2_segmentation_results.png')
