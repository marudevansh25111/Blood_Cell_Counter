"""
Image Preprocessing Module for Blood Cell Counter

This module handles all image preprocessing operations including:
- Image loading and validation
- Noise reduction using bilateral filtering
- Contrast enhancement with CLAHE
- Color space conversions
- Morphological operations

Optimized for Mac M3 Pro performance.
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImagePreprocessor:
    """
    Handles all image preprocessing operations for blood cell analysis.
    
    This class provides a complete preprocessing pipeline that transforms
    raw microscopic blood images into processed images suitable for cell
    detection and classification.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default parameters."""
        self.processed_image = None
        self.original_image = None
        self.preprocessing_steps = []
    
    def load_image(self, image_path):
        """
        Load image from file path with error handling.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image in RGB format, or None if failed
        """
        try:
            # Load image using OpenCV
            self.original_image = cv2.imread(image_path)
            
            if self.original_image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB for consistency
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            print(f"✅ Image loaded successfully: {self.original_image.shape}")
            return self.original_image
        
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def preprocess_image(self, image=None):
        """
        Complete preprocessing pipeline for blood cell images.
        
        Args:
            image (numpy.ndarray, optional): Input image. Uses loaded image if None.
            
        Returns:
            numpy.ndarray: Processed grayscale image ready for cell detection
        """
        if image is None:
            image = self.original_image
        
        if image is None:
            raise ValueError("No image to process. Load an image first.")
        
        print("Starting image preprocessing pipeline...")
        
        # Step 1: Noise reduction
        print("  1. Reducing noise...")
        denoised = self.reduce_noise(image)
        self.preprocessing_steps.append(('denoised', denoised))
        
        # Step 2: Enhance contrast
        print("  2. Enhancing contrast...")
        enhanced = self.enhance_contrast(denoised)
        self.preprocessing_steps.append(('enhanced', enhanced))
        
        # Step 3: Color space conversion
        print("  3. Converting color space...")
        hsv_image = self.convert_color_space(enhanced)
        self.preprocessing_steps.append(('hsv', hsv_image))
        
        # Step 4: Morphological operations
        print("  4. Applying morphological operations...")
        self.processed_image = self.apply_morphology(hsv_image)
        self.preprocessing_steps.append(('final', self.processed_image))
        
        print("✅ Preprocessing complete!")
        return self.processed_image
    
    def reduce_noise(self, image):
        """
        Apply bilateral filtering to reduce noise while preserving edges.
        
        Bilateral filtering is particularly effective for medical images as it
        smooths noise while maintaining sharp cell boundaries.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        # Parameters optimized for blood cell images
        d = 9          # Diameter of pixel neighborhood
        sigma_color = 75   # Filter sigma in color space
        sigma_space = 75   # Filter sigma in coordinate space
        
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return denoised
    
    def enhance_contrast(self, image):
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE improves local contrast and is particularly effective for
        medical images with varying illumination conditions.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Contrast-enhanced image
        """
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L (lightness) channel only
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    def convert_color_space(self, image):
        """
        Convert image to HSV color space for better color-based analysis.
        
        HSV (Hue, Saturation, Value) color space is more intuitive for
        color-based cell classification than RGB.
        
        Args:
            image (numpy.ndarray): Input RGB image
            
        Returns:
            numpy.ndarray: HSV image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return hsv
    
    def apply_morphology(self, image):
        """
        Apply morphological operations to clean up the image.
        
        Operations include:
        - Opening: Remove small noise
        - Closing: Fill small gaps
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Processed grayscale image
        """
        # Convert to grayscale for morphological operations
        gray = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        
        # Create elliptical kernel (better for circular cells)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Opening operation (erosion followed by dilation)
        # Removes small noise and separates connected objects
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Closing operation (dilation followed by erosion)
        # Fills small gaps and smooths object contours
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return closed
    
    def visualize_preprocessing_steps(self):
        """
        Create visualization of all preprocessing steps.
        
        Displays a 2x3 grid showing the original image and each
        preprocessing step for analysis and debugging.
        """
        if self.original_image is None or self.processed_image is None:
            print("❌ No images to visualize. Process an image first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Blood Cell Image Preprocessing Pipeline', fontsize=16, fontweight='bold')
        
        # Original image
        axes[0, 0].imshow(self.original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Show preprocessing steps
        step_titles = ['Noise Reduced', 'Contrast Enhanced', 'HSV Color Space', 'Final Processed']
        positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
        
        for i, ((pos_row, pos_col), title) in enumerate(zip(positions, step_titles)):
            if i < len(self.preprocessing_steps):
                step_name, step_image = self.preprocessing_steps[i]
                
                if step_name == 'final':
                    # Show grayscale image
                    axes[pos_row, pos_col].imshow(step_image, cmap='gray')
                else:
                    # Show color image
                    axes[pos_row, pos_col].imshow(step_image)
                
                axes[pos_row, pos_col].set_title(title)
                axes[pos_row, pos_col].axis('off')
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_preprocessing_stats(self):
        """
        Get statistics about the preprocessing results.
        
        Returns:
            dict: Dictionary containing preprocessing statistics
        """
        if self.original_image is None or self.processed_image is None:
            return None
        
        stats = {
            'original_shape': self.original_image.shape,
            'processed_shape': self.processed_image.shape,
            'original_dtype': str(self.original_image.dtype),
            'processed_dtype': str(self.processed_image.dtype),
            'original_range': (self.original_image.min(), self.original_image.max()),
            'processed_range': (self.processed_image.min(), self.processed_image.max()),
            'preprocessing_steps_count': len(self.preprocessing_steps)
        }
        
        return stats
    
    def save_processed_image(self, output_path):
        """
        Save the processed image to disk.
        
        Args:
            output_path (str): Path where to save the processed image
        """
        if self.processed_image is None:
            print("❌ No processed image to save.")
            return False
        
        try:
            cv2.imwrite(output_path, self.processed_image)
            print(f"✅ Processed image saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return False