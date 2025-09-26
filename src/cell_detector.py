"""
Cell Detection Module for Blood Cell Counter

This module implements multiple algorithms for detecting blood cells:
- HoughCircles for circular cells (RBCs)
- Contour detection for irregular shapes
- Watershed segmentation for overlapping cells

Combines multiple detection methods for robust cell identification.
"""

import cv2
import numpy as np
from skimage import segmentation, measure
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

class CellDetector:
    """
    Detects blood cells using multiple computer vision algorithms.
    
    This class implements a multi-algorithm approach to cell detection,
    combining different methods to achieve robust and accurate detection
    of various blood cell types.
    """
    
    def __init__(self):
        """Initialize the cell detector with default parameters."""
        self.detected_cells = []
        self.detection_image = None
        self.detection_methods_used = []
    
    def detect_cells(self, processed_image, original_image):
        """
        Main cell detection pipeline using multiple algorithms.
        
        Args:
            processed_image (numpy.ndarray): Preprocessed grayscale image
            original_image (numpy.ndarray): Original RGB image for region extraction
            
        Returns:
            list: List of detected cell regions with metadata
        """
        print("Starting cell detection pipeline...")
        self.detection_image = processed_image.copy()
        
        # Method 1: HoughCircles for round cells (primarily RBCs)
        print("  1. Detecting circular cells with HoughCircles...")
        round_cells = self.detect_round_cells(processed_image)
        print(f"     Found {len(round_cells)} circular cells")
        
        # Method 2: Contour detection for all shapes
        print("  2. Detecting cells with contour analysis...")
        contour_cells = self.detect_contour_cells(processed_image)
        print(f"     Found {len(contour_cells)} contour-based cells")
        
        # Method 3: Watershed segmentation for overlapping cells
        print("  3. Applying watershed segmentation...")
        watershed_cells = self.watershed_segmentation(processed_image)
        print(f"     Found {len(watershed_cells)} watershed-separated cells")
        
        # Combine and filter detections
        print("  4. Combining and filtering detections...")
        all_cells = self.combine_detections(round_cells, contour_cells, watershed_cells)
        print(f"     Combined to {len(all_cells)} unique cells")
        
        # Extract cell regions from original image
        print("  5. Extracting cell regions...")
        cell_regions = self.extract_cell_regions(all_cells, original_image)
        
        self.detected_cells = cell_regions
        print(f"✅ Detection complete! Found {len(cell_regions)} valid cell regions")
        
        return cell_regions
    
    def detect_round_cells(self, image):
        """
        Detect circular cells using HoughCircles algorithm.
        
        Particularly effective for Red Blood Cells (RBCs) which are
        typically circular in shape.
        
        Args:
            image (numpy.ndarray): Processed grayscale image
            
        Returns:
            list: List of detected circular cells with metadata
        """
        # HoughCircles parameters optimized for blood cells
        circles = cv2.HoughCircles(
            image,
            cv2.HOUGH_GRADIENT,    # Detection method
            dp=1,                   # Inverse ratio of accumulator resolution
            minDist=30,            # Minimum distance between circle centers
            param1=50,             # Upper threshold for edge detection
            param2=30,             # Accumulator threshold for center detection
            minRadius=10,          # Minimum circle radius
            maxRadius=50           # Maximum circle radius
        )
        
        detected_circles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Validate circle is within image bounds
                if (x - r >= 0 and y - r >= 0 and 
                    x + r < image.shape[1] and y + r < image.shape[0]):
                    
                    detected_circles.append({
                        'center': (x, y),
                        'radius': r,
                        'method': 'hough_circle',
                        'bbox': (x - r, y - r, x + r, y + r),
                        'confidence': 0.8  # High confidence for circular detection
                    })
        
        return detected_circles
    
    def detect_contour_cells(self, image):
        """
        Detect cells using contour analysis.
        
        Effective for detecting cells of various shapes including
        irregular White Blood Cells (WBCs) and Platelets.
        
        Args:
            image (numpy.ndarray): Processed grayscale image
            
        Returns:
            list: List of detected cells with contour information
        """
        # Apply Otsu thresholding for optimal binary conversion
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, 
            cv2.RETR_EXTERNAL,      # Retrieve only outer contours
            cv2.CHAIN_APPROX_SIMPLE # Compress contours to save memory
        )
        
        detected_contours = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Filter by area to remove noise and too-large objects
            if 100 < area < 8000:  # Adjusted range for blood cells
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate additional shape features
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate circularity (4π × Area / Perimeter²)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Calculate aspect ratio
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                detected_contours.append({
                    'center': (center_x, center_y),
                    'contour': contour,
                    'area': area,
                    'perimeter': perimeter,
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'method': 'contour',
                    'bbox': (x, y, x + w, y + h),
                    'confidence': min(0.9, circularity + 0.1)  # Higher confidence for circular shapes
                })
        
        return detected_contours
    
    def watershed_segmentation(self, image):
        """
        Apply watershed algorithm to separate overlapping cells.
        
        Particularly useful for detecting cells that are touching or
        overlapping in the image.
        
        Args:
            image (numpy.ndarray): Processed grayscale image
            
        Returns:
            list: List of watershed-separated cell regions
        """
        # Apply Otsu thresholding
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Distance transform to find cell centers
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find local maxima as potential cell centers
        # local_maxima = peak_local_maxima(
        #     dist_transform,
        #     min_distance=20,                    # Minimum distance between peaks
        #     threshold_abs=0.3 * dist_transform.max(),  # Minimum peak height
        #     exclude_border=True                 # Exclude border peaks
        # )
        # Simple peak detection
        threshold = 0.3 * dist_transform.max()
        local_maxima_mask = (maximum_filter(dist_transform, size=20) == dist_transform) & (dist_transform > threshold)
        local_maxima = np.where(local_maxima_mask)
        local_maxima = list(zip(local_maxima[0], local_maxima[1]))
        
        # Create markers for watershed
        markers = np.zeros(image.shape, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed algorithm
        labels = segmentation.watershed(-dist_transform, markers, mask=binary > 0)
        
        # Extract regions from watershed result
        watershed_cells = []
        
        for region in measure.regionprops(labels):
            # Filter by area
            if region.area > 150:  # Minimum area for valid cells
                
                # Get region properties
                y, x = region.centroid
                bbox = region.bbox  # (min_row, min_col, max_row, max_col)
                
                # Calculate additional properties
                eccentricity = region.eccentricity
                solidity = region.solidity
                
                watershed_cells.append({
                    'center': (int(x), int(y)),
                    'area': region.area,
                    'eccentricity': eccentricity,
                    'solidity': solidity,
                    'method': 'watershed',
                    'bbox': bbox,
                    'label': region.label,
                    'confidence': min(0.7, solidity)  # Confidence based on shape solidity
                })
        
        return watershed_cells
    
    def combine_detections(self, round_cells, contour_cells, watershed_cells):
        """
        Combine detections from multiple methods and remove duplicates.
        
        Uses proximity-based filtering to eliminate duplicate detections
        of the same cell by different algorithms.
        
        Args:
            round_cells (list): Detections from HoughCircles
            contour_cells (list): Detections from contour analysis
            watershed_cells (list): Detections from watershed segmentation
            
        Returns:
            list: Combined and filtered list of unique detections
        """
        # Combine all detections
        all_detections = round_cells + contour_cells + watershed_cells
        
        # Track which methods were used
        methods_used = set()
        for detection in all_detections:
            methods_used.add(detection['method'])
        
        self.detection_methods_used = list(methods_used)
        
        # Remove duplicates based on proximity
        filtered_detections = []
        duplicate_threshold = 25  # pixels
        
        for detection in all_detections:
            is_duplicate = False
            
            for existing in filtered_detections:
                # Calculate Euclidean distance between centers
                dist = np.sqrt(
                    (detection['center'][0] - existing['center'][0]) ** 2 +
                    (detection['center'][1] - existing['center'][1]) ** 2
                )
                
                if dist < duplicate_threshold:
                    is_duplicate = True
                    
                    # Keep the detection with higher confidence
                    if detection.get('confidence', 0) > existing.get('confidence', 0):
                        filtered_detections.remove(existing)
                        filtered_detections.append(detection)
                    
                    break
            
            if not is_duplicate:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def extract_cell_regions(self, detections, original_image):
        """
        Extract cell image regions from original image based on detections.
        
        Args:
            detections (list): List of cell detections with bounding boxes
            original_image (numpy.ndarray): Original RGB image
            
        Returns:
            list: List of cell regions with extracted image patches
        """
        cell_regions = []
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            
            # Handle different bbox formats
            if len(bbox) == 4:
                # Format: (x1, y1, x2, y2) or (min_row, min_col, max_row, max_col)
                if detection.get('method') == 'watershed':
                    # Watershed format: (min_row, min_col, max_row, max_col)
                    y1, x1, y2, x2 = bbox
                else:
                    # Other formats: (x1, y1, x2, y2)
                    x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within image bounds
                h, w = original_image.shape[:2]
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                
                # Extract region if valid
                if x2 > x1 and y2 > y1:
                    cell_region = original_image[y1:y2, x1:x2]
                    
                    # Only add if region has sufficient size
                    if cell_region.size > 0:
                        cell_regions.append({
                            'id': i,
                            'image': cell_region,
                            'bbox': (x1, y1, x2, y2),
                            'center': detection['center'],
                            'method': detection['method'],
                            'area': (x2 - x1) * (y2 - y1),
                            'confidence': detection.get('confidence', 0.5)
                        })
        
        return cell_regions
    
    def visualize_detections(self, original_image):
        """
        Create visualization of detected cells with bounding boxes and labels.
        
        Args:
            original_image (numpy.ndarray): Original RGB image
            
        Returns:
            numpy.ndarray: Annotated image with detection visualizations
        """
        if not self.detected_cells:
            print("❌ No cells detected to visualize")
            return None
        
        # Create copy for visualization
        viz_image = original_image.copy()
        
        # Color mapping for different detection methods
        method_colors = {
            'hough_circle': (255, 0, 0),    # Red
            'contour': (0, 255, 0),         # Green
            'watershed': (0, 0, 255),       # Blue
            'combined': (255, 255, 0)       # Yellow
        }
        
        # Draw detections
        for cell in self.detected_cells:
            bbox = cell['bbox']
            center = cell['center']
            method = cell.get('method', 'unknown')
            confidence = cell.get('confidence', 0.0)
            
            # Get color for this method
            color = method_colors.get(method, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw center point
            cv2.circle(viz_image, center, 3, color, -1)
            
            # Add text label with ID and confidence
            label = f"ID:{cell['id']} ({confidence:.2f})"
            label_pos = (bbox[0], bbox[1] - 10)
            
            # Add background for text readability
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            cv2.rectangle(
                viz_image, 
                (label_pos[0], label_pos[1] - text_height - 2),
                (label_pos[0] + text_width, label_pos[1] + 2),
                color, -1
            )
            
            # Add text
            cv2.putText(
                viz_image, label, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        
        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(viz_image)
        plt.title(f'Blood Cell Detection Results\nDetected: {len(self.detected_cells)} cells')
        plt.axis('off')
        
        # Add legend for methods
        legend_elements = []
        for method in self.detection_methods_used:
            color = [c/255.0 for c in method_colors.get(method, (128, 128, 128))]
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, label=method))
        
        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
        
        return viz_image
    
    def get_detection_statistics(self):
        """
        Get comprehensive statistics about the detection process.
        
        Returns:
            dict: Dictionary containing detection statistics
        """
        if not self.detected_cells:
            return None
        
        # Count detections by method
        method_counts = {}
        confidence_scores = []
        areas = []
        
        for cell in self.detected_cells:
            method = cell.get('method', 'unknown')
            confidence = cell.get('confidence', 0.0)
            area = cell.get('area', 0)
            
            method_counts[method] = method_counts.get(method, 0) + 1
            confidence_scores.append(confidence)
            areas.append(area)
        
        stats = {
            'total_cells_detected': len(self.detected_cells),
            'detection_methods_used': self.detection_methods_used,
            'detections_by_method': method_counts,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_range': (min(confidence_scores), max(confidence_scores)) if confidence_scores else (0, 0),
            'average_cell_area': np.mean(areas) if areas else 0,
            'area_range': (min(areas), max(areas)) if areas else (0, 0),
            'detection_density': len(self.detected_cells) / (self.detection_image.shape[0] * self.detection_image.shape[1]) if self.detection_image is not None else 0
        }
        
        return stats
    
    def filter_detections_by_confidence(self, min_confidence=0.3):
        """
        Filter detected cells by minimum confidence threshold.
        
        Args:
            min_confidence (float): Minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            list: Filtered list of high-confidence detections
        """
        if not self.detected_cells:
            return []
        
        filtered_cells = [
            cell for cell in self.detected_cells 
            if cell.get('confidence', 0.0) >= min_confidence
        ]
        
        print(f"Filtered {len(self.detected_cells)} cells to {len(filtered_cells)} "
              f"with confidence >= {min_confidence}")
        
        return filtered_cells
    
    def save_detections(self, output_path):
        """
        Save detection results to file.
        
        Args:
            output_path (str): Path to save detection results
        """
        if not self.detected_cells:
            print("❌ No detections to save")
            return False
        
        try:
            import json
            
            # Prepare data for JSON serialization
            detections_data = []
            for cell in self.detected_cells:
                cell_data = {
                    'id': cell['id'],
                    'center': cell['center'],
                    'bbox': cell['bbox'],
                    'method': cell['method'],
                    'area': cell['area'],
                    'confidence': cell.get('confidence', 0.0)
                }
                detections_data.append(cell_data)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump({
                    'total_detections': len(detections_data),
                    'detection_methods_used': self.detection_methods_used,
                    'detections': detections_data
                }, f, indent=2)
            
            print(f"✅ Detections saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving detections: {e}")
            return False