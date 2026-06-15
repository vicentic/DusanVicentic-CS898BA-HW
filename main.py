import cv2
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
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

total_images = base_images + transformed_images
total_info = base_names.copy()

height, width = img. shape[:2]
center = (width / 2, height / 2)

for i, image in enumerate(base_images):
    for j in range(2):
        angle = random.randint(1, 359)
        scale = random.uniform(0.5, 1.5)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        warped = cv2.warpAffine(image, matrix, (width, height))
        transformed_images.append(warped)
        total_info.append(f'{base_names[i]}_affine_{j+1}')
        cv2.imwrite(f'{output_dir}/{base_names[i]}_affine_{j+1}.jpg', warped)

# print(len(total_images))

#Applying Gaussian blur
sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
blurred_images = []
blurred_info = []

for image in total_images:
    for s in sigmas:
        blur = cv2.GaussianBlur(image, (0, 0), sigmaX=s)
        
        blurred_images.append(blur)
        blurred_info.append(f'{total_info[i]}_blurred_{s}')
        cv2.imwrite(f'{output_dir}/{total_info[i]}_blurred_{s}.jpg', blur)

all_images = total_images + blurred_images
all_info = total_info + blurred_info

# Create random subset

zipped_168 = list(zip(all_images, all_info))
subset = random.sample(zipped_168, 42)

readme_indices = random.sample(range(42), 6)
plots_saved = 1

# Greyscale images for edge detection
for i, (image, history_text) in enumerate(subset):
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # Edge detection
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    laplacian_8u = cv2.convertScaleAbs(laplacian)
    
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_8u = cv2.convertScaleAbs(sobel_combined)

    canny = cv2.Canny(gray_img, 100, 200)

    kernerlx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernerly = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_x = cv2.filter2D(gray_img, -1, kernerlx)
    prewitt_y = cv2.filter2D(gray_img, -1, kernerly)
    prewitt_combined = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

    # Save images
    cv2.imwrite(f'{output_dir}/10_subset_input_{i+1}.jpg', image)
    cv2.imwrite(f'{output_dir}/10_subset_laplacian_{i+1}.jpg', laplacian_8u)
    cv2.imwrite(f'{output_dir}/10_subset_sobel_{i+1}.jpg', sobel_8u)
    cv2.imwrite(f'{output_dir}/10_subset_canny_{i+1}.jpg', canny)
    cv2.imwrite(f'{output_dir}/10_subset_prewitt_{i+1}.jpg', prewitt_combined)  

    # Generate plots
    images_to_plot = [image, laplacian_8u, sobel_8u, canny, prewitt_combined]
    titles = ['Input', 'Laplacian', 'Sobel', 'Canny', 'Prewitt']
        
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    fig.suptitle(f'Processing History: {history_text}', fontsize=16, fontweight='bold')
        
    for k in range(5):
        img_display = images_to_plot[k]
        if k == 0 and len(img_display.shape) == 3:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        axes[k].imshow(img_display, cmap='gray')
        axes[k].set_title(titles[k])
        axes[k].axis('off')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_subset_plots_{i+1}.png')

    if i in readme_indices:
        plt.savefig(f'{output_dir}/README_plot_{plots_saved}.png')        
        plots_saved += 1

    plt.close(fig)

print(f'I am officially bored with this')
