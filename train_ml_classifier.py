#!/usr/bin/env python3
"""
Machine Learning Training Script for Blood Cell Counter

This script trains a Random Forest classifier on real blood cell images.
"""

import os
import sys
import cv2
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append('src')
from cell_classifier import CellClassifier

class MLTrainer:
    def __init__(self):
        self.classifier_module = CellClassifier()
        self.scaler = StandardScaler()
        self.trained_model = None
        
    def load_images_from_folder(self, folder_path, label):
        """Load and extract features from images in a folder"""
        features = []
        labels = []
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist")
            return features, labels
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        
        print(f"Processing {len(image_files)} {label} images...")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(folder_path, img_file)
            
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extract features using your existing feature extraction
                cell_features = self.classifier_module.extract_features(image_rgb)
                
                if cell_features is not None:
                    features.append(cell_features)
                    labels.append(label)
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
        
        print(f"Successfully extracted features from {len(features)} {label} images")
        return features, labels
    
    def load_dataset(self, dataset_path='data/real_dataset'):
        """Load complete dataset and extract features"""
        print("Loading dataset and extracting features...")
        
        all_features = []
        all_labels = []
        
        # Load training data
        for cell_type in ['RBC', 'WBC', 'Platelet']:
            train_folder = os.path.join(dataset_path, 'train', cell_type)
            features, labels = self.load_images_from_folder(train_folder, cell_type)
            all_features.extend(features)
            all_labels.extend(labels)
        
        if len(all_features) == 0:
            raise ValueError("No training data found! Please organize your dataset properly.")
        
        return np.array(all_features), np.array(all_labels)
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("Training Random Forest classifier...")
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        )
        
        rf_model.fit(X_train, y_train)
        return rf_model
    
    def train_svm(self, X_train, y_train):
        """Train SVM classifier"""
        print("Training SVM classifier...")
        
        svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42,
            class_weight='balanced'
        )
        
        svm_model.fit(X_train, y_train)
        return svm_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        print(f"\nEvaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['RBC', 'WBC', 'Platelet'],
                   yticklabels=['RBC', 'WBC', 'Platelet'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        os.makedirs('data/output/training_results', exist_ok=True)
        plt.savefig(f'data/output/training_results/{model_name.lower()}_confusion_matrix.png')
        plt.close()
        
        # Calculate accuracy
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
    
    def cross_validate_model(self, model, X, y, model_name):
        """Perform cross-validation"""
        print(f"\nPerforming cross-validation for {model_name}...")
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        print(f"{model_name} Cross-validation scores: {cv_scores}")
        print(f"{model_name} Average CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_scores
    
    def save_model(self, model, model_name, scaler, training_stats):
        """Save trained model and metadata"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'model_name': model_name,
            'feature_names': self.classifier_module.feature_names,
            'cell_types': self.classifier_module.cell_types,
            'training_timestamp': datetime.now().isoformat(),
            'training_stats': training_stats
        }
        
        model_path = f'models/{model_name.lower()}_blood_cell_classifier.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def train_and_compare_models(self):
        """Main training function - trains and compares multiple models"""
        
        print("="*60)
        print("BLOOD CELL COUNTER - ML TRAINING")
        print("="*60)
        
        # Load dataset
        try:
            features, labels = self.load_dataset()
            print(f"\nDataset loaded successfully!")
            print(f"Total samples: {len(features)}")
            print(f"Feature dimensions: {features.shape[1]}")
            
            # Print class distribution
            unique, counts = np.unique(labels, return_counts=True)
            print(f"\nClass distribution:")
            for class_name, count in zip(unique, counts):
                print(f"  {class_name}: {count} samples")
        
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nPlease ensure you have:")
            print("1. Downloaded a blood cell dataset")
            print("2. Organized images in data/real_dataset/train/{RBC,WBC,Platelet}/")
            return False
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train models
        models = {}
        results = {}
        
        # Train Random Forest
        models['RandomForest'] = self.train_random_forest(X_train, y_train)
        results['RandomForest'] = self.evaluate_model(
            models['RandomForest'], X_test, y_test, 'RandomForest'
        )
        
        # Train SVM
        models['SVM'] = self.train_svm(X_train, y_train)
        results['SVM'] = self.evaluate_model(
            models['SVM'], X_test, y_test, 'SVM'
        )
        
        # Cross-validation
        for model_name, model in models.items():
            cv_scores = self.cross_validate_model(model, features_scaled, labels, model_name)
            results[model_name]['cv_scores'] = cv_scores.tolist()
            results[model_name]['cv_mean'] = cv_scores.mean()
            results[model_name]['cv_std'] = cv_scores.std()
        
        # Compare results
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        best_model_name = None
        best_accuracy = 0
        
        for model_name, result in results.items():
            accuracy = result['accuracy']
            cv_mean = result['cv_mean']
            
            print(f"\n{model_name}:")
            print(f"  Test Accuracy: {accuracy:.3f}")
            print(f"  CV Accuracy: {cv_mean:.3f} (+/- {result['cv_std']:.3f})")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        print(f"\nBest model: {best_model_name} (accuracy: {best_accuracy:.3f})")
        
        # Save best model
        best_model = models[best_model_name]
        model_path = self.save_model(
            best_model, 
            best_model_name, 
            self.scaler,
            results[best_model_name]
        )
        
        # Save training report
        training_report = {
            'training_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(features),
                'feature_dimensions': features.shape[1],
                'class_distribution': dict(zip(unique, counts.tolist()))
            },
            'model_results': results,
            'best_model': best_model_name,
            'model_path': model_path
        }
        
        report_path = 'data/output/training_results/training_report.json'
        with open(report_path, 'w') as f:
            json.dump(training_report, f, indent=2)
        
        print(f"\nTraining report saved to: {report_path}")
        
        # Update the main classifier to use the trained model
        print(f"\nUpdating main classifier to use trained {best_model_name} model...")
        self.classifier_module.load_classifier(model_path)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Best model: {best_model_name}")
        print(f"Test accuracy: {best_accuracy:.3f}")
        print(f"Model saved to: {model_path}")
        
        return True

def main():
    """Main function"""
    trainer = MLTrainer()
    
    # Check if dataset exists
    if not os.path.exists('data/real_dataset/train'):
        print("Dataset not found!")
        print("\nPlease follow these steps:")
        print("1. Download blood cell dataset from Kaggle")
        print("2. Create folder: data/real_dataset/train/{RBC,WBC,Platelet}/")
        print("3. Copy images to respective folders")
        print("4. Run this script again")
        return False
    
    # Start training
    success = trainer.train_and_compare_models()
    
    if success:
        print("\nNext steps:")
        print("1. Test the trained model: python test_implementation.py")
        print("2. Use ML classification in web app by checking 'Use ML Classification'")
        print("3. Compare performance with rule-based classification")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)