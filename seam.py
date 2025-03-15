# Step 1: Install Required Libraries
!pip install numpy opencv-python matplotlib

# Step 2: Import Necessary Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 3: Load and Display Image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def show_image(image, title="Image"):
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()

# Step 4: Compute Energy Function
# Manually compute gradient magnitude (e1 energy)
def compute_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = np.abs(np.roll(gray, -1, axis=1) - np.roll(gray, 1, axis=1))
    dy = np.abs(np.roll(gray, -1, axis=0) - np.roll(gray, 1, axis=0))
    return dx + dy

# Step 5: Find the Optimal Seam using Dynamic Programming
def find_seam(energy):
    h, w = energy.shape
    seam_map = energy.copy()
    backtrack = np.zeros_like(energy, dtype=int)

    for i in range(1, h):
        for j in range(w):
            left = seam_map[i-1, j-1] if j > 0 else float('inf')
            middle = seam_map[i-1, j]
            right = seam_map[i-1, j+1] if j < w - 1 else float('inf')
            
            min_energy = min(left, middle, right)
            backtrack[i, j] = j - 1 if min_energy == left else j + 1 if min_energy == right else j
            seam_map[i, j] += min_energy
    
    return seam_map, backtrack

# Step 6: Remove the Seam
def remove_seam(img, backtrack):
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    j = np.argmin(backtrack[-1])
    
    for i in reversed(range(h)):
        mask[i, j] = False
        j = backtrack[i, j]
    
    img = img[mask].reshape((h, w - 1, 3))
    return img

# Step 7: Seam Carving to Reduce Image Size
def seam_carving(img, num_seams, direction="horizontal"):
    for _ in range(num_seams):
        energy_map = compute_energy(img)
        seam_map, backtrack = find_seam(energy_map)
        img = remove_seam(img, backtrack)
    return img

# Step 8: Visualize Removed Seams
def visualize_seams(img, num_seams):
    seam_img = img.copy()
    for _ in range(num_seams):
        energy_map = compute_energy(seam_img)
        seam_map, backtrack = find_seam(energy_map)
        
        j = np.argmin(seam_map[-1])
        for i in reversed(range(seam_img.shape[0])):
            seam_img[i, j] = [255, 0, 0]  # Mark seam in red
            j = backtrack[i, j]
        
        seam_img = remove_seam(seam_img, backtrack)
    return seam_img

# Performance Optimization
def optimized_seam_carving(img, num_seams):
    img = img.astype(np.float32)
    for _ in range(num_seams):
        energy_map = compute_energy(img.astype(np.uint8))
        seam_map, backtrack = find_seam(energy_map)
        img = remove_seam(img, backtrack)
    return img.astype(np.uint8)

# Step 9: Run Seam Carving Example
image_path = "path of image inserted here "  # Use uploaded image
image = load_image(image_path)
num_seams = image.shape[1] // 4  # Reduce width by 25%

# Compute energy map
energy_map = compute_energy(image)
show_image(energy_map, "Energy Map")

# Visualize seams before removal
seam_visualization = visualize_seams(image, num_seams)
show_image(seam_visualization, "Seam Visualization")

# Apply optimized seam carving
resized_image = optimized_seam_carving(image, num_seams)
show_image(resized_image, "Resized Image")
