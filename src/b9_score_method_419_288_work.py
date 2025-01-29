import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple

def display_images(imgs, titles=None, show_debug_images=True):
    """
    Helper function to display multiple images side by side
    """
    if not show_debug_images:
        return
        
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

def parse_filename(filename):
    """Extract ground truth coordinates from filename"""
    parts = filename.split('-')
    if len(parts) < 3:
        return None
    
    try:
        # Get bounding box coordinates
        bbox = parts[2].split('_')
        if len(bbox) != 2:
            return None
            
        # Parse coordinates
        top_left, bottom_right = bbox[0].split('&'), bbox[1].split('&')
        x1, y1 = map(int, top_left)
        x2, y2 = map(int, bottom_right)
        
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
    except:
        return None

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes"""
    # Create binary masks for both boxes
    h, w = max(np.max(box1[:, 1]), np.max(box2[:, 1])), max(np.max(box1[:, 0]), np.max(box2[:, 0]))
    mask1 = np.zeros((h+1, w+1), dtype=np.uint8)
    mask2 = np.zeros((h+1, w+1), dtype=np.uint8)
    
    # Fill masks
    cv2.fillPoly(mask1, [box1], 1)
    cv2.fillPoly(mask2, [box2], 1)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0

def detect_plate(image_path, show_debug_images=True, debug_stats=None):
    if debug_stats is None:
        debug_stats = {
            'total_contours': 0,
            'area_ratio_filtered': 0,
            'vertex_count_filtered': 0,
            'angle_filtered': 0,
            'aspect_ratio_filtered': 0
        }
    
    gt_box = parse_filename(os.path.basename(image_path))
    img = cv2.imread(image_path)
    img_area = img.shape[0] * img.shape[1]
    location = None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to HSV and enhance green channel
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Multiple green ranges for different shades
    masks = []
    
    # Bright green
    lower_green1 = np.array([40, 50, 50])
    upper_green1 = np.array([90, 255, 255])
    green_mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    masks.append(green_mask1)
    
    # Dark green
    lower_green2 = np.array([40, 50, 20])
    upper_green2 = np.array([90, 255, 150])
    green_mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    masks.append(green_mask2)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(green_mask1, green_mask2)
    
    # Enhance mask
    kernel = np.ones((3,3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    potential_plates = []
    
    for contour in contours:
        debug_stats['total_contours'] += 1
        area = cv2.contourArea(contour)
        area_ratio = area / img_area
        
        # More permissive area ratio
        if area_ratio < 0.005 or area_ratio > 0.08:
            debug_stats['area_ratio_filtered'] += 1
            continue
        
        # Use convex hull
        hull = cv2.convexHull(contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # More permissive vertex count
        if not (4 <= len(approx) <= 8):
            debug_stats['vertex_count_filtered'] += 1
            continue
        
        rect = cv2.minAreaRect(contour)
        (center, (width, height), angle) = rect
        
        # Normalize angle
        if angle < -45:
            angle += 90
            width, height = height, width
        
        # More permissive angle
        if not (-20 <= angle <= 20):
            debug_stats['angle_filtered'] += 1
            continue
        
        aspect_ratio = max(width, height) / min(width, height)
        # More permissive aspect ratio
        if not (2.0 <= aspect_ratio <= 5.0):
            debug_stats['aspect_ratio_filtered'] += 1
            continue
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # Scoring system
        score = 0
        
        # 1. Green color density score
        roi_mask = np.zeros_like(combined_mask)
        cv2.drawContours(roi_mask, [box], 0, 255, -1)
        green_density = cv2.countNonZero(cv2.bitwise_and(combined_mask, roi_mask)) / cv2.countNonZero(roi_mask)
        score += green_density * 4.0  # Higher weight for green color
        
        # 2. Aspect ratio score
        aspect_score = 1 - abs(aspect_ratio - 3.5) / 2.0
        score += aspect_score * 2.0
        
        # 3. Area score
        optimal_area_ratio = 0.02
        area_score = 1 - abs(area_ratio - optimal_area_ratio) / 0.02
        score += area_score * 2.0
        
        # 4. Position score (prefer middle of image)
        cy = center[1] / img.shape[0]
        position_score = 1.0 - abs(cy - 0.5) * 2
        score += position_score * 1.0
        
        if score > 4.0:  # More permissive threshold
            potential_plates.append((box, score))
    
    # Select best candidate
    if potential_plates:
        potential_plates.sort(key=lambda x: x[1], reverse=True)
        location = potential_plates[0][0]
    
    if location is None:
        return None, gt_box, None, debug_stats
        
    # Rest of the visualization code remains the same
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)
    
    # Crop the license plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2+1, y1:y2+1]
    
    # Modified visualization code
    result = img.copy()
    
    # Draw detected plate in green
    cv2.drawContours(result, [location], 0, (0, 255, 0), 2)
    
    # Draw ground truth in orange if available
    if gt_box is not None:
        cv2.drawContours(result, [gt_box], 0, (0, 165, 255), 2)
        
        # Calculate and display IoU
        iou = calculate_iou(location, gt_box)
        cv2.putText(result, f'IoU: {iou:.2f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Modified visualization code
    debug_images = [
        img,
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR),  # Changed from edge_mask to combined_mask
        new_image,
        cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR),
        result
    ]
    debug_titles = ['Original', 'Grayscale', 'Green Mask', 
                   'Plate Region', 'Cropped Plate', 'Detection vs Ground Truth']
    display_images(debug_images, debug_titles, show_debug_images)
    
    return location, gt_box, iou if gt_box is not None else None, debug_stats

# Set up paths and process images
base_path = '../dataset/CCPD2020/ccpd_green'
dataset_path = f'{base_path}/train/images'
sample_images = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]

# Create a named tuple for storing detection results
DetectionResult = namedtuple('DetectionResult', ['detection', 'ground_truth', 'iou'])

def process_single_image(img_name, dataset_path, show_debug_images):
    """Process a single image and return its detection results"""
    print(f"\nProcessing {img_name}")
    img_path = os.path.join(dataset_path, img_name)
    detection, ground_truth, iou, debug_stats = detect_plate(img_path, show_debug_images)
    return DetectionResult(detection, ground_truth, iou), debug_stats

# Modified main loop with threading
total_images = 0
detected_plates = 0
good_detections = 0
iou_scores = []

# Use max_workers based on CPU count or a fixed number
show_debug_images = False  # Set this to False to disable image display

# Initialize counters for the filtering statistics
total_stats = {
    'total_contours': 0,
    'area_ratio_filtered': 0,
    'vertex_count_filtered': 0,
    'angle_filtered': 0,
    'aspect_ratio_filtered': 0
}

with ThreadPoolExecutor() as executor:
    # Submit all tasks
    future_to_img = {
        executor.submit(process_single_image, img_name, dataset_path, show_debug_images): img_name 
        for img_name in sample_images
    }
    
    # Process results as they complete
    for future in future_to_img:
        try:
            result, stats = future.result()
            total_images += 1
            
            # Accumulate statistics
            for key in total_stats:
                total_stats[key] += stats[key]
                
            if result.detection is not None:
                detected_plates += 1
                if result.iou is not None:
                    iou_scores.append(result.iou)
                    if result.iou > 0.5:
                        good_detections += 1
        except Exception as e:
            print(f"Error processing {future_to_img[future]}: {str(e)}")

# Print statistics
print("\nDetection Statistics:")
print(f"Total images processed: {total_images}")
if total_images > 0:  # Add check for division by zero
    print(f"Plates detected: {detected_plates} ({detected_plates/total_images*100:.1f}%)")
    print(f"Good detections (IoU > 0.5): {good_detections} ({good_detections/total_images*100:.1f}%)")
    if iou_scores:
        print(f"Average IoU: {np.mean(iou_scores):.3f}")
else:
    print("No images were processed successfully")

print("\nFiltering Statistics:")
print(f"Total contours examined: {total_stats['total_contours']}")
print(f"Filtered by area ratio: {total_stats['area_ratio_filtered']}")
print(f"Filtered by vertex count: {total_stats['vertex_count_filtered']}")
print(f"Filtered by angle: {total_stats['angle_filtered']}")
print(f"Filtered by aspect ratio: {total_stats['aspect_ratio_filtered']}")
