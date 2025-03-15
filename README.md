# Seam-Carving
This project implements seam carving to resize images without cropping, preserving important content while removing less significant regions. The algorithm removes low-energy seams iteratively to reduce the width or height of an image.

# Features

Compute Energy Map: Uses the gradient magnitude method to determine important regions.
Find Optimal Seam: Uses dynamic programming to locate the lowest-energy seam.
Remove Seams Iteratively: Removes seams to reduce the image size.
Seam Visualization: Highlights removed seams in red before removal.
Bonus Optimization: Implements an optimized version for better performance.

# Installation
Run the following command in Google Colab to install the required dependencies:

!pip install numpy opencv-python matplotlib

# How to Run
Upload an image to Google Colab (e.g., /content/lake.jpg).
Run the provided Python script to:
Compute the energy map.
Visualize removed seams.
Resize the image while preserving content.
Adjust the number of seams removed by modifying:

num_seams = image.shape[1] // 4  # Reduce width by 25%

# Output Examples
The program will generate:
Original Image
Energy Map
Seam Visualization (red lines showing removed seams)
Resized Image


# Author
Sohyla Said
Developed as part of the CSE 429: Computer Vision and Pattern Recognition course under Dr. Ahmed Gomaa.
