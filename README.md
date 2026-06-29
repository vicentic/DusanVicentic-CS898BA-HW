# CS898BA - HW 1

## Instructions

1. open project directory and create Python virtual environment
'python3 -m venv venv'

2. activate environment
'source venv/bin/activate (Mac) or '.\venv\Scripts\activate' (Windows)

3. install dependencies:
'pip install opencv-python numpy scipy matplotlib'

4. run scipt:
(making sure image is in directory)

## Gausian Blur

Applying a gausian blur before edge detection reduces noise. At higher sigmas, the picture smoothes out too much which causes missed edges. At lower sigmas, pictures are too noisy resulting in noise being picked up as an edge

## Edge Detection

Used 4 different algorithms on heavily modified pictures.

1. Sobel: first derivative changes, provides clear boundaries resistant to small noise.
2. Prewitt: similar to sobel, without center weighting. Weaker edges, and more artifacts than sobel
3. Laplacian: Second derivative, too sensitive, chaotic edge maps if Gausian blur was not strong enoug
4. Canny: Best results, but highes computation cost. Provided continous, sharp outlines of objects

## Plots

![Plot 1](./output_images/README_plot_1.png)
![Plot 2](./output_images/README_plot_2.png)
![Plot 3](./output_images/README_plot_3.png)
![Plot 4](./output_images/README_plot_4.png)
![Plot 5](./output_images/README_plot_5.png)
![Plot 6](./output_images/README_plot_6.png)