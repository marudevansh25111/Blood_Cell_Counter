#!/usr/bin/env python3
"""
WBC-Specific Training Script for Blood Cell Counter

This script handles the case where we only have WBC images.
Creates a binary classifier (WBC vs non-WBC).
"""

import os
import sys
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from datetime import datetime

# Add src to path
sys.path.append('src')
from cell_classifier import CellClassifier

class WBCTrainer:
    def __init__(self):
        self.classifier_module = CellClassifier()
        self.trained_model = None
        
    def load_wbc_images(self, folder_path='data/real_dataset/train/WBC'):
        """Load WBC images and extract features"""
        features = []
        
        if not os.path.exists(folder_path):
            raise ValueError(f"WBC folder not found: {folder_path}")
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        print(f"Loading {len(image_files)} WBC images...")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract features
                cell_features = self.classifier_module.extract_features(image_rgb)
                
                if cell_features is not None:
                    features.append(cell_features)
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        print(f"Successfully extracted features from {len(features)} WBC images")
        return np.array(features)
    
    def create_synthetic_non_wbc_features(self, wbc_features, num_samples=2000):
        """Create synthetic non-WBC features by modifying WBC features"""
        print(f"Creating {num_samples} synthetic non-WBC samples...")
        
        non_wbc_features = []
        
        for i in range(num_samples):
            # Take a random WBC feature vector
            base_features = wbc_features[np.random.randint(0, len(wbc_features))].copy()
            
            # Modify features to simulate RBC characteristics
            if i < num_samples // 2:  # Half as RBC-like
                # RBCs are typically more circular and red
                base_features[2] = min(0.9, base_features[2] + np.random.normal(0.1, 0.05))  # Higher circularity
                base_features[4] = min(255, base_features[4] + np.random.normal(30, 10))      # More red
                base_features[5] = max(50, base_features[5] - np.random.normal(20, 10))       # Less green
                base_features[6] = max(50, base_features[6] - np.random.normal(20, 10))       # Less blue
                base_features[0] = max(500, min(8000, base_features[0] * np.random.uniform(0.3, 0.8)))  # Different size
            
            else:  # Half as Platelet-like
                # Platelets are smaller and more irregular
                base_features[2] = max(0.2, base_features[2] - np.random.normal(0.2, 0.05))  # Lower circularity
                base_features[0] = max(100, base_features[0] * np.random.uniform(0.1, 0.4))  # Much smaller
                base_features[4] = min(255, base_features[4] + np.random.normal(20, 10))     # Lighter
                base_features[5] = min(255, base_features[5] + np.random.normal(20, 10))
                base_features[6] = min(255, base_features[6] + np.random.normal(20, 10))
            
            non_wbc_features.append(base_features)
        
        return np.array(non_wbc_features)
    
    def train_binary_classifier(self):
        """Train a binary WBC vs non-WBC classifier"""
        
        print("="*60)
        print("WBC BINARY CLASSIFIER TRAINING")
        print("="*60)
        
        # Load WBC features
        wbc_features = self.load_wbc_images()
        
        if len(wbc_features) == 0:
            raise ValueError("No WBC features extracted!")
        
        # Create synthetic non-WBC features
        non_wbc_features = self.create_synthetic_non_wbc_features(wbc_features)
        
        # Combine features and labels
        all_features = np.vstack([wbc_features, non_wbc_features])
        all_labels = np.array(['WBC'] * len(wbc_features) + ['non-WBC'] * len(non_wbc_features))
        
        print(f"\nDataset created:")
        print(f"WBC samples: {len(wbc_features)}")
        print(f"Non-WBC samples: {len(non_wbc_features)}")
        print(f"Total samples: {len(all_features)}")
        print(f"Feature dimensions: {all_features.shape[1]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            all_features, all_labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=all_labels
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest classifier
        print("\nTraining Random Forest binary classifier...")
        
        self.trained_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.trained_model.fit(X_train, y_train)
        
        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.trained_model.predict(X_test)
        
        print("\nBinary Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate accuracy
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print(f"Test Accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def create_hybrid_classifier(self):
        """Create a hybrid classifier that combines ML WBC detection with rule-based RBC/Platelet"""
        
        class HybridClassifier:
            def __init__(self, wbc_model, feature_names, cell_types):
                self.wbc_model = wbc_model
                self.feature_names = feature_names
                self.cell_types = cell_types
                self.rule_classifier = CellClassifier()
            
            def predict(self, features_array):
                """Predict using hybrid approach"""
                predictions = []
                
                for features in features_array:
                    # First, check if it's a WBC using ML
                    wbc_prob = self.wbc_model.predict_proba([features])[0]
                    wbc_confidence = max(wbc_prob)
                    is_wbc = self.wbc_model.predict([features])[0] == 'WBC'
                    
                    if is_wbc and wbc_confidence > 0.7:
                        predictions.append('WBC')
                    else:
                        # Use rule-based for non-WBC classification
                        rule_prediction = self.rule_classifier.rule_based_classification(features)
                        predictions.append(rule_prediction)
                
                return np.array(predictions)
            
            def predict_proba(self, features_array):
                """Return prediction probabilities"""
                probabilities = []
                
                for features in features_array:
                    wbc_prob = self.wbc_model.predict_proba([features])[0]
                    is_wbc = self.wbc_model.predict([features])[0] == 'WBC'
                    
                    if is_wbc:
                        # Return WBC probability
                        probabilities.append([0.1, max(wbc_prob), 0.1, 1-max(wbc_prob)])  # [RBC, WBC, Platelet, Unknown]
                    else:
                        # Rule-based confidence
                        probabilities.append([0.8, 0.1, 0.8, 0.3])  # Default rule-based confidence
                
                return np.array(probabilities)
        
        hybrid_model = HybridClassifier(
            self.trained_model, 
            self.classifier_module.feature_names,
            ['RBC', 'WBC', 'Platelet', 'Unknown']
        )
        
        return hybrid_model
    
    def save_model(self, accuracy):
        """Save the trained hybrid model"""
        
        # Create hybrid classifier
        hybrid_model = self.create_hybrid_classifier()
        
        model_data = {
            'model': hybrid_model,
            'model_type': 'hybrid_wbc_classifier',
            'feature_names': self.classifier_module.feature_names,
            'cell_types': ['RBC', 'WBC', 'Platelet', 'Unknown'],
            'training_timestamp': datetime.now().isoformat(),
            'wbc_training_accuracy': accuracy,
            'description': 'Hybrid classifier: ML for WBC detection, rules for RBC/Platelet'
        }
        
        model_path = 'models/hybrid_wbc_blood_cell_classifier.pkl'
        os.makedirs('models', exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Hybrid model saved to: {model_path}")
        
        # Update main classifier
        print("Updating main classifier to use hybrid model...")
        self.classifier_module.load_classifier(model_path)
        
        return model_path
    
    def train_complete_system(self):
        """Complete training pipeline"""
        
        try:
            # Train binary WBC classifier
            accuracy = self.train_binary_classifier()
            
            # Save hybrid model
            model_path = self.save_model(accuracy)
            
            # Generate report
            training_report = {
                'training_timestamp': datetime.now().isoformat(),
                'model_type': 'hybrid_wbc_classifier',
                'wbc_accuracy': accuracy,
                'model_path': model_path,
                'description': 'Trained on WBC images with synthetic non-WBC data'
            }
            
            os.makedirs('data/output/training_results', exist_ok=True)
            report_path = 'data/output/training_results/wbc_training_report.json'
            
            with open(report_path, 'w') as f:
                json.dump(training_report, f, indent=2)
            
            print(f"\n{'='*60}")
            print("HYBRID WBC TRAINING COMPLETED!")
            print(f"{'='*60}")
            print(f"WBC Detection Accuracy: {accuracy:.3f}")
            print(f"Model Type: Hybrid (ML for WBC, Rules for RBC/Platelet)")
            print(f"Model saved to: {model_path}")
            print(f"Training report: {report_path}")
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main training function"""
    
    trainer = WBCTrainer()
    
    # Check if WBC data exists
    wbc_folder = 'data/real_dataset/train/WBC'
    if not os.path.exists(wbc_folder):
        print("WBC training data not found!")
        print(f"Expected folder: {wbc_folder}")
        return False
    
    wbc_count = len([f for f in os.listdir(wbc_folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if wbc_count == 0:
        print("No WBC images found!")
        return False
    
    print(f"Found {wbc_count} WBC images for training")
    
    # Start training
    success = trainer.train_complete_system()
    
    if success:
        print("\nNext steps:")
        print("1. Test the hybrid model: python test_implementation.py")
        print("2. Use in web app with 'Use ML Classification' enabled")
        print("3. Compare with rule-based classification")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)