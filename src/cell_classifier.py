"""
Cell Classification Module for Blood Cell Counter

This module implements classification algorithms for blood cells:
- Feature extraction from cell images
- Rule-based classification using biological characteristics  
- Machine learning classification using Random Forest
- Performance evaluation and reporting

Classifies cells into: RBC, WBC, Platelet, Unknown
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

class CellClassifier:
    """
    Classifies blood cells using feature extraction and multiple classification methods.
    
    This class provides both rule-based and machine learning approaches
    for classifying detected blood cells into different types based on
    morphological and color characteristics.
    """
    
    def __init__(self):
        """Initialize the classifier with default parameters."""
        self.classifier = None
        self.feature_names = [
            'area', 'perimeter', 'circularity', 'aspect_ratio',
            'mean_red', 'mean_green', 'mean_blue',
            'mean_hue', 'mean_saturation', 'mean_value',
            'std_red', 'std_green', 'std_blue'
        ]
        self.cell_types = ['RBC', 'WBC', 'Platelet', 'Unknown']
    
    def extract_features(self, cell_image):
        """
        Extract comprehensive features from a cell image.
        
        Features include:
        - Morphological: area, perimeter, circularity, aspect ratio
        - Color (RGB): mean and standard deviation for each channel
        - Color (HSV): mean values for hue, saturation, value
        
        Args:
            cell_image (numpy.ndarray): Cell image region (RGB)
            
        Returns:
            numpy.ndarray: Feature vector, or None if extraction fails
        """
        if cell_image is None or cell_image.size == 0:
            return None
        
        try:
            # Ensure image is in correct format
            if len(cell_image.shape) != 3 or cell_image.shape[2] != 3:
                return None
            
            # Convert to different color spaces
            hsv_image = cv2.cvtColor(cell_image, cv2.COLOR_RGB2HSV)
            gray_image = cv2.cvtColor(cell_image, cv2.COLOR_RGB2GRAY)
            
            # Basic morphological features
            height, width = gray_image.shape
            area = height * width
            
            # Find contours for shape analysis
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate shape features
            if contours:
                # Get the largest contour (main cell body)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Calculate perimeter
                perimeter = cv2.arcLength(largest_contour, True)
                
                # Calculate circularity (4π × Area / Perimeter²)
                if perimeter > 0:
                    contour_area = cv2.contourArea(largest_contour)
                    circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                else:
                    circularity = 0
                    perimeter = 0
                
                # Calculate aspect ratio from bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 1.0
                
            else:
                # Default values if no contours found
                perimeter = 2 * (height + width)  # Rectangle perimeter as estimate
                circularity = 0
                aspect_ratio = float(width) / height if height > 0 else 1.0
            
            # Color features - RGB channels
            mean_red = np.mean(cell_image[:, :, 0])
            mean_green = np.mean(cell_image[:, :, 1])
            mean_blue = np.mean(cell_image[:, :, 2])
            
            std_red = np.std(cell_image[:, :, 0])
            std_green = np.std(cell_image[:, :, 1])
            std_blue = np.std(cell_image[:, :, 2])
            
            # Color features - HSV channels
            mean_hue = np.mean(hsv_image[:, :, 0])
            mean_saturation = np.mean(hsv_image[:, :, 1])
            mean_value = np.mean(hsv_image[:, :, 2])
            
            # Combine all features into feature vector
            features = np.array([
                area, perimeter, circularity, aspect_ratio,
                mean_red, mean_green, mean_blue,
                mean_hue, mean_saturation, mean_value,
                std_red, std_green, std_blue
            ])
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def rule_based_classification(self, features):
        """
        Classify cell using rule-based approach based on biological characteristics.
        
        Classification rules:
        - RBC: Circular, red-pink color, medium size, no nucleus
        - WBC: Larger, irregular, contains dark nucleus, varied shapes
        - Platelet: Very small, irregular fragments, light color
        
        Args:
            features (numpy.ndarray): Feature vector from extract_features()
            
        Returns:
            str: Cell type ('RBC', 'WBC', 'Platelet', or 'Unknown')
        """
        if features is None or len(features) != len(self.feature_names):
            return 'Unknown'
        
        # Extract feature values
        area, perimeter, circularity, aspect_ratio = features[0:4]
        mean_red, mean_green, mean_blue = features[4:7]
        mean_hue, mean_saturation, mean_value = features[7:10]
        std_red, std_green, std_blue = features[10:13]
        
        # RBC Classification Rules
        # Characteristics: Circular shape, red-pink color, medium size
        if (circularity > 0.65 and                              # High circularity
            aspect_ratio > 0.6 and aspect_ratio < 1.4 and      # Nearly square aspect ratio
            mean_red > mean_blue and mean_red > mean_green and  # Reddish color
            mean_saturation > 50 and                            # Sufficient color saturation
            1000 < area < 12000 and                             # Medium size range
            std_red < 40):                                      # Uniform color
            return 'RBC'
        
        # WBC Classification Rules  
        # Characteristics: Larger size, irregular shape, contains nucleus (darker regions)
        elif (area > 5000 and                                   # Large size
              circularity < 0.85 and                           # Less circular
              mean_value < 200 and                             # Generally darker (nucleus)
              std_red > 20 or std_green > 20 or std_blue > 20): # Color variation (nucleus vs cytoplasm)
            return 'WBC'
        
        # Platelet Classification Rules
        # Characteristics: Very small, irregular fragments, light color
        elif (area < 3000 and                                   # Small size
              circularity < 0.7 and                            # Irregular shape
              mean_value > 150):                                # Generally lighter
            return 'Platelet'
        
        # If none of the rules match
        return 'Unknown'
    
    def train_ml_classifier(self, training_data, labels):
        """
        Train machine learning classifier using Random Forest.
        
        Args:
            training_data (numpy.ndarray): Feature matrix (n_samples, n_features)
            labels (numpy.ndarray): Target labels
            
        Returns:
            sklearn.ensemble.RandomForestClassifier: Trained classifier
        """
        if len(training_data) == 0 or len(labels) == 0:
            print("❌ No training data provided")
            return None
        
        print(f"Training ML classifier with {len(training_data)} samples...")
        
        # Initialize Random Forest with optimized parameters
        self.classifier = RandomForestClassifier(
            n_estimators=200,           # Number of trees
            max_depth=15,              # Maximum depth of trees
            min_samples_split=5,       # Minimum samples to split node
            min_samples_leaf=2,        # Minimum samples in leaf
            random_state=42,           # Reproducible results
            class_weight='balanced'    # Handle class imbalance
        )
        
        # Train the classifier
        self.classifier.fit(training_data, labels)
        
        print("✅ ML Classifier trained successfully!")
        
        # Print feature importance
        feature_importance = self.classifier.feature_importances_
        for i, (feature, importance) in enumerate(zip(self.feature_names, feature_importance)):
            if importance > 0.05:  # Show only important features
                print(f"  {feature}: {importance:.3f}")
        
        return self.classifier
    
    def classify_cell(self, cell_image, use_ml=False):
        """
        Classify a single cell image.
        
        Args:
            cell_image (numpy.ndarray): Cell image region
            use_ml (bool): Whether to use ML classifier (if trained)
            
        Returns:
            tuple: (cell_type, confidence_score)
        """
        # Extract features
        features = self.extract_features(cell_image)
        
        if features is None:
            return 'Unknown', 0.0
        
        if use_ml and self.classifier is not None:
            # Use ML classifier
            try:
                prediction = self.classifier.predict([features])[0]
                confidence = max(self.classifier.predict_proba([features])[0])
                return prediction, confidence
            except Exception as e:
                print(f"ML classification failed: {e}")
                # Fall back to rule-based
                prediction = self.rule_based_classification(features)
                confidence = 0.6 if prediction != 'Unknown' else 0.1
                return prediction, confidence
        else:
            # Use rule-based classifier
            prediction = self.rule_based_classification(features)
            
            # Assign confidence based on how well rules were satisfied
            if prediction == 'Unknown':
                confidence = 0.1
            else:
                # Higher confidence for clear classifications
                area, _, circularity, aspect_ratio = features[0:4]
                
                if prediction == 'RBC' and circularity > 0.8:
                    confidence = 0.9
                elif prediction == 'WBC' and area > 8000:
                    confidence = 0.85
                elif prediction == 'Platelet' and area < 1500:
                    confidence = 0.8
                else:
                    confidence = 0.7
            
            return prediction, confidence
    
    def classify_all_cells(self, cell_regions, use_ml=False):
        """
        Classify all detected cell regions.
        
        Args:
            cell_regions (list): List of cell regions from detector
            use_ml (bool): Whether to use ML classifier
            
        Returns:
            list: List of classification results with metadata
        """
        print(f"Classifying {len(cell_regions)} cells...")
        
        classifications = []
        
        for i, cell in enumerate(cell_regions):
            cell_type, confidence = self.classify_cell(cell['image'], use_ml)
            
            classifications.append({
                'id': cell['id'],
                'type': cell_type,
                'confidence': confidence,
                'bbox': cell['bbox'],
                'center': cell['center'],
                'area': cell['area'],
                'detection_method': cell.get('method', 'unknown')
            })
            
            # Progress indicator for large batches
            if len(cell_regions) > 20 and (i + 1) % 10 == 0:
                print(f"  Classified {i + 1}/{len(cell_regions)} cells...")
        
        print("✅ Classification complete!")
        return classifications
    
    def generate_report(self, classifications):
        """
        Generate comprehensive classification report.
        
        Args:
            classifications (list): List of classification results
            
        Returns:
            dict: Detailed report with counts, percentages, and statistics
        """
        if not classifications:
            return {
                'total_cells': 0,
                'counts': {cell_type: 0 for cell_type in self.cell_types},
                'percentages': {cell_type: 0.0 for cell_type in self.cell_types},
                'average_confidence': 0.0,
                'classifications': []
            }
        
        # Count each cell type
        counts = {cell_type: 0 for cell_type in self.cell_types}
        confidence_scores = []
        confidence_by_type = {cell_type: [] for cell_type in self.cell_types}
        
        for classification in classifications:
            cell_type = classification['type']
            confidence = classification['confidence']
            
            counts[cell_type] += 1
            confidence_scores.append(confidence)
            
            if cell_type in confidence_by_type:
                confidence_by_type[cell_type].append(confidence)
        
        total_cells = len(classifications)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Calculate percentages
        percentages = {
            cell_type: (count / total_cells * 100) if total_cells > 0 else 0.0 
            for cell_type, count in counts.items()
        }
        
        # Calculate confidence statistics by type
        confidence_stats = {}
        for cell_type, confs in confidence_by_type.items():
            if confs:
                confidence_stats[cell_type] = {
                    'mean': np.mean(confs),
                    'std': np.std(confs),
                    'min': min(confs),
                    'max': max(confs)
                }
            else:
                confidence_stats[cell_type] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0
                }
        
        report = {
            'total_cells': total_cells,
            'counts': counts,
            'percentages': percentages,
            'average_confidence': avg_confidence,
            'confidence_by_type': confidence_stats,
            'classifications': classifications
        }
        
        return report
    
    def evaluate_classification(self, true_labels, predicted_labels):
        """
        Evaluate classification performance against ground truth.
        
        Args:
            true_labels (list): True cell type labels
            predicted_labels (list): Predicted cell type labels
            
        Returns:
            dict: Evaluation metrics including accuracy, precision, recall
        """
        if len(true_labels) != len(predicted_labels):
            print("❌ Mismatch in label lengths")
            return None
        
        # Calculate classification report
        report = classification_report(
            true_labels, predicted_labels, 
            target_names=self.cell_types,
            output_dict=True
        )
        
        # Calculate confusion matrix
        conf_matrix = confusion_matrix(
            true_labels, predicted_labels,
            labels=self.cell_types
        )
        
        evaluation = {
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'overall_accuracy': report['accuracy'],
            'macro_avg_f1': report['macro avg']['f1-score'],
            'weighted_avg_f1': report['weighted avg']['f1-score']
        }
        
        return evaluation
    
    def save_classifier(self, filepath):
        """
        Save trained classifier to disk.
        
        Args:
            filepath (str): Path to save the classifier
        """
        if self.classifier is None:
            print("❌ No trained classifier to save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save classifier and metadata
            classifier_data = {
                'classifier': self.classifier,
                'feature_names': self.feature_names,
                'cell_types': self.cell_types
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(classifier_data, f)
            
            print(f"✅ Classifier saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving classifier: {e}")
            return False
    
    def load_classifier(self, filepath):
        """
        Load trained classifier from disk.
        
        Args:
            filepath (str): Path to the saved classifier
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            with open(filepath, 'rb') as f:
                classifier_data = pickle.load(f)
            
            self.classifier = classifier_data['classifier']
            self.feature_names = classifier_data.get('feature_names', self.feature_names)
            self.cell_types = classifier_data.get('cell_types', self.cell_types)
            
            print(f"✅ Classifier loaded from: {filepath}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Classifier file not found: {filepath}")
            return False
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            return False
    
    def get_classification_summary(self, report):
        """
        Get human-readable summary of classification results.
        
        Args:
            report (dict): Report from generate_report()
            
        Returns:
            str: Formatted summary string
        """
        if not report or report['total_cells'] == 0:
            return "No cells classified."
        
        summary_lines = []
        summary_lines.append(f"CLASSIFICATION SUMMARY")
        summary_lines.append("=" * 30)
        summary_lines.append(f"Total Cells: {report['total_cells']}")
        summary_lines.append(f"Average Confidence: {report['average_confidence']:.3f}")
        summary_lines.append("")
        summary_lines.append("Cell Type Distribution:")
        
        for cell_type in self.cell_types:
            count = report['counts'][cell_type]
            percentage = report['percentages'][cell_type]
            if count > 0:
                conf_stats = report['confidence_by_type'][cell_type]
                summary_lines.append(
                    f"  {cell_type:>8}: {count:3d} cells ({percentage:5.1f}%) "
                    f"[conf: {conf_stats['mean']:.3f}±{conf_stats['std']:.3f}]"
                )
        
        return "\n".join(summary_lines)