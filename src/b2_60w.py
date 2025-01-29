import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

base_path = '../dataset/CCPD2020/ccpd_green'

dataset_path = f'{base_path}/test/images'
sample_images = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))][:20]

SCALE_FACTOR = 1 # Factor that scales down the image resolution

def display_images(imgs, titles=None):
    """
    Helper function to display multiple images side by side
    """
    plt.figure(figsize=(20, 4))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        # Convert BGR to RGB for display
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
    plt.tight_layout()
    plt.show()

# Preprocess images

def preprocess_image(image_path):
    # Read and resize image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (int(img.shape[1] * SCALE_FACTOR), int(img.shape[0] * SCALE_FACTOR)))
    
    # Create copies for different processing methods
    edge_img = img.copy()
    color_img = img.copy()
    
    # 1. Edge Detection Approach
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(edge_img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Dilate edges to connect them
    kernel = np.ones((5,5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)
    
    # 2. Color-Based Segmentation Approach
    # Convert to HSV
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    
    # Define multiple color ranges for vehicles
    color_ranges = [
        # White vehicles
        (np.array([0, 0, 200]), np.array([180, 30, 255])),
        # Light gray/silver vehicles
        (np.array([0, 0, 140]), np.array([180, 30, 200])),
        # Dark vehicles
        (np.array([0, 0, 60]), np.array([180, 60, 140]))
    ]
    
    # Create combined color mask
    color_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for lower, upper in color_ranges:
        mask = cv2.inRange(hsv, lower, upper)
        color_mask = cv2.bitwise_or(color_mask, mask)
    
    # Clean up color mask
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. Combine both approaches
    combined_mask = cv2.bitwise_and(dilated_edges, color_mask)
    
    # 4. Post-processing
    # Apply morphological operations to clean up the mask
    kernel_large = np.ones((20,20), np.uint8)
    processed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter contours by area
        h, w = img.shape[:2]
        min_area = (h * w) * 0.05
        max_area = (h * w) * 0.8
        valid_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
        
        if valid_contours:
            # Create mask from largest contour
            car_mask = np.zeros((h, w), dtype=np.uint8)
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(car_mask, [largest_contour], -1, (255, 255, 255), -1)
            
            # Apply mask to original image
            result = cv2.bitwise_and(img, img, mask=car_mask)
            
            # Debug visualizations
            debug_images = [
                img,
                cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR),  # Convert to BGR for display
                cv2.bitwise_and(img, img, mask=color_mask),
                cv2.bitwise_and(img, img, mask=combined_mask),
                cv2.bitwise_and(img, img, mask=processed),
                result
            ]
            debug_titles = [
                'Original',
                'Edge Detection',
                'Color Segmentation',
                'Combined Mask',
                'Processed Mask',
                'Result'
            ]
            display_images(debug_images, debug_titles)
            
            return result
    
    return img

def detect_license_plate(image_path, debug=True):
    """
    License plate detection supporting both blue and green plates with improved filtering
    """
    # Read and resize image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    max_dimension = 800
    scale = max_dimension / max(height, width)
    if scale < 1:
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color ranges for both blue and green plates
    color_ranges = [
        # Blue plates
        (np.array([100, 80, 80]), np.array([130, 255, 255])),
        # Green plates
        (np.array([35, 80, 80]), np.array([85, 255, 255]))
    ]

    # Combine masks for both colors
    combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for lower, upper in color_ranges:
        color_mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, color_mask)

    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    plate_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    plate_mask = cv2.morphologyEx(plate_mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours with improved criteria
    candidates = []
    img_area = img.shape[0] * img.shape[1]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by relative area (0.1% to 5% of image)
        if not (0.001 * img_area < area < 0.05 * img_area):
            continue
            
        # Get rotated rectangle
        rect = cv2.minAreaRect(contour)
        center, (width, height), angle = rect
        
        # Scale up the rectangle by 20%
        scale_factor = 1.2
        width *= scale_factor
        height *= scale_factor
        
        # Create new scaled rectangle
        rect = (center, (width, height), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Get width and height for aspect ratio check
        if width < height:
            width, height = height, width
            
        # Filter by aspect ratio (Chinese plates are typically 3.14:1)
        aspect_ratio = width / height if height != 0 else 0
        if not (2.8 <= aspect_ratio <= 3.5):
            continue
            
        # Filter by solidity (area ratio between contour and its convex hull)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area != 0 else 0
        if solidity < 0.8:  # Expect plate to be mostly rectangular
            continue

        candidates.append(contour)

    if debug and candidates:
        debug_img = img.copy()
        cropped_plate = None
        
        # Get the largest candidate (assuming it's the most likely plate)
        if candidates:
            largest_contour = max(candidates, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            center, (width, height), angle = rect
            width *= 1.2  # Scale up by 20%
            height *= 1.3
            scaled_rect = (center, (width, height), angle)
            box = cv2.boxPoints(scaled_rect)
            box = np.int0(box)
            
            # Draw all candidates
            for contour in candidates:
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect
                width *= 1.2
                height *= 1.3
                scaled_rect = (center, (width, height), angle)
                box = cv2.boxPoints(scaled_rect)
                box = np.int0(box)
                cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
            
            # Extract the plate region for the largest candidate
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])
            
            src_pts = box.astype("float32")
            # Coordinate the points so the plate is oriented correctly
            dst_pts = np.array([[0, height-1],
                              [0, 0],
                              [width-1, 0],
                              [width-1, height-1]], dtype="float32")
            
            # Get the perspective transform matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # Perform the perspective transform
            cropped_plate = cv2.warpPerspective(img, M, (width, height))
        
        debug_images = [
            img,
            cv2.bitwise_and(img, img, mask=plate_mask),
            debug_img,
            cropped_plate if cropped_plate is not None else np.zeros_like(img)
        ]
        debug_titles = ['Original', 'Color Mask', 'Detected Plates', 'Cropped Plate']
        display_images(debug_images, debug_titles)

    return candidates, img

# Process images
for img_name in sample_images:
    img_path = os.path.join(dataset_path, img_name)
    candidates, img = detect_license_plate(img_path)
    if candidates:
        print(f"Found {len(candidates)} potential license plates in {img_name}")
