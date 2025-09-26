#!/usr/bin/env python3
"""
Dataset organization script for your specific blood cell dataset structure
"""

import os
import shutil
import pandas as pd
from pathlib import Path

def organize_dataset(base_path):
    """Organize your specific dataset structure"""
    
    print("Organizing your blood cell dataset...")
    print(f"Base path: {base_path}")
    
    # Create target structure
    target_base = "data/real_dataset"
    os.makedirs(f"{target_base}/train/RBC", exist_ok=True)
    os.makedirs(f"{target_base}/train/WBC", exist_ok=True)
    os.makedirs(f"{target_base}/train/Platelet", exist_ok=True)
    os.makedirs(f"{target_base}/test/RBC", exist_ok=True)
    os.makedirs(f"{target_base}/test/WBC", exist_ok=True)
    os.makedirs(f"{target_base}/test/Platelet", exist_ok=True)
    
    # Path to dataset2-master (has the organized WBC images)
    dataset2_path = os.path.join(base_path, "dataset2-master", "dataset2-master", "images")
    
    # Path to dataset-master (has JPEGImages and Annotations)
    dataset1_path = os.path.join(base_path, "dataset-master", "dataset-master")
    
    # Check if paths exist
    if not os.path.exists(dataset2_path):
        print(f"❌ Could not find: {dataset2_path}")
        return False
    
    print(f"✅ Found dataset2 at: {dataset2_path}")
    
    # Organize WBC images from dataset2-master
    wbc_mapping = {
        'EOSINOPHIL': 'WBC',
        'LYMPHOCYTE': 'WBC',
        'MONOCYTE': 'WBC',
        'NEUTROPHIL': 'WBC'
    }
    
    copied_counts = {'train': {'WBC': 0}, 'test': {'WBC': 0}}
    
    # Copy WBC training images
    train_path = os.path.join(dataset2_path, "TRAIN")
    if os.path.exists(train_path):
        print("Copying WBC training images...")
        for wbc_type, target_type in wbc_mapping.items():
            source_folder = os.path.join(train_path, wbc_type)
            if os.path.exists(source_folder):
                target_folder = os.path.join(target_base, "train", target_type)
                
                for img_file in os.listdir(source_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_file = os.path.join(source_folder, img_file)
                        target_file = os.path.join(target_folder, f"{wbc_type}_{img_file}")
                        shutil.copy2(source_file, target_file)
                        copied_counts['train']['WBC'] += 1
                
                print(f"  Copied {wbc_type} images")
    
    # Copy WBC test images
    test_path = os.path.join(dataset2_path, "TEST")
    if os.path.exists(test_path):
        print("Copying WBC test images...")
        for wbc_type, target_type in wbc_mapping.items():
            source_folder = os.path.join(test_path, wbc_type)
            if os.path.exists(source_folder):
                target_folder = os.path.join(target_base, "test", target_type)
                
                for img_file in os.listdir(source_folder):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        source_file = os.path.join(source_folder, img_file)
                        target_file = os.path.join(target_folder, f"{wbc_type}_{img_file}")
                        shutil.copy2(source_file, target_file)
                        copied_counts['test']['WBC'] += 1
    
    # Try to process dataset-master for RBC and Platelets using labels.csv
    labels_csv = os.path.join(base_path, "labels.csv")
    jpeg_images_path = os.path.join(dataset1_path, "JPEGImages")
    
    if os.path.exists(labels_csv) and os.path.exists(jpeg_images_path):
        print("Processing dataset-master with labels.csv...")
        try:
            # Read labels
            df = pd.read_csv(labels_csv)
            print(f"Loaded {len(df)} labels from CSV")
            print(f"Label columns: {df.columns.tolist()}")
            print(f"Sample labels:\n{df.head()}")
            
            # Try to identify label columns (common names)
            label_column = None
            filename_column = None
            
            for col in df.columns:
                if 'label' in col.lower() or 'class' in col.lower() or 'category' in col.lower():
                    label_column = col
                if 'file' in col.lower() or 'image' in col.lower() or 'name' in col.lower():
                    filename_column = col
            
            if label_column and filename_column:
                print(f"Using label column: {label_column}")
                print(f"Using filename column: {filename_column}")
                
                # Map labels to our categories
                label_mapping = {
                    'rbc': 'RBC', 'red': 'RBC', 'erythrocyte': 'RBC',
                    'wbc': 'WBC', 'white': 'WBC', 'leukocyte': 'WBC',
                    'platelet': 'Platelet', 'thrombocyte': 'Platelet'
                }
                
                copied_counts['train']['RBC'] = 0
                copied_counts['train']['Platelet'] = 0
                
                for _, row in df.iterrows():
                    filename = row[filename_column]
                    label = str(row[label_column]).lower()
                    
                    # Find matching cell type
                    target_type = None
                    for key, value in label_mapping.items():
                        if key in label:
                            target_type = value
                            break
                    
                    if target_type and target_type in ['RBC', 'Platelet']:
                        source_file = os.path.join(jpeg_images_path, filename)
                        if os.path.exists(source_file):
                            target_folder = os.path.join(target_base, "train", target_type)
                            target_file = os.path.join(target_folder, filename)
                            shutil.copy2(source_file, target_file)
                            copied_counts['train'][target_type] += 1
            
        except Exception as e:
            print(f"Error processing labels.csv: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET ORGANIZATION SUMMARY")
    print("="*50)
    
    for split in ['train', 'test']:
        print(f"\n{split.upper()} SET:")
        for cell_type in ['RBC', 'WBC', 'Platelet']:
            folder_path = os.path.join(target_base, split, cell_type)
            if os.path.exists(folder_path):
                count = len([f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"  {cell_type}: {count} images")
            else:
                print(f"  {cell_type}: 0 images")
    
    # Check if we have enough data for training
    total_train_images = sum([
        len([f for f in os.listdir(os.path.join(target_base, "train", cell_type)) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for cell_type in ['RBC', 'WBC', 'Platelet']
        if os.path.exists(os.path.join(target_base, "train", cell_type))
    ])
    
    print(f"\nTotal training images: {total_train_images}")
    
    if total_train_images > 100:
        print("✅ Sufficient data for ML training!")
        print("Next step: run 'python train_ml_classifier.py'")
        return True
    else:
        print("⚠️  Limited training data. Consider:")
        print("1. Adding more images manually")
        print("2. Using synthetic data generation")
        print("3. Using rule-based classification (current system)")
        return False

def main():
    # Use the path you provided
    base_path = "/Users/devanshmaru/Downloads/blood_cell_data"
    
    print("Blood Cell Dataset Organizer")
    print("="*40)
    print(f"Processing dataset at: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ Base path does not exist: {base_path}")
        return
    
    success = organize_dataset(base_path)
    
    if success:
        print("\n🎉 Dataset organization completed successfully!")
    else:
        print("\n⚠️  Dataset organized but may need additional setup")

if __name__ == "__main__":
    main()