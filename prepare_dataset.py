#!/usr/bin/env python3
"""
Dataset Preparation Script

Helps organize downloaded blood cell images into the correct structure.
"""

import os
import shutil
from pathlib import Path

def organize_kaggle_dataset(source_path, target_path='data/real_dataset'):
    """
    Organize Kaggle blood cell dataset into train/test structure
    
    Expected Kaggle structure:
    dataset2-master/
    ├── dataset2-master/images/
    │   ├── TRAIN/
    │   │   ├── EOSINOPHIL/
    │   │   ├── LYMPHOCYTE/
    │   │   ├── MONOCYTE/
    │   │   └── NEUTROPHIL/
    │   └── TEST/
    │       ├── EOSINOPHIL/
    │       ├── LYMPHOCYTE/
    │       ├── MONOCYTE/
    │       └── NEUTROPHIL/
    """
    
    # Create target structure
    os.makedirs(f'{target_path}/train/RBC', exist_ok=True)
    os.makedirs(f'{target_path}/train/WBC', exist_ok=True)
    os.makedirs(f'{target_path}/train/Platelet', exist_ok=True)
    os.makedirs(f'{target_path}/test/RBC', exist_ok=True)
    os.makedirs(f'{target_path}/test/WBC', exist_ok=True)
    os.makedirs(f'{target_path}/test/Platelet', exist_ok=True)
    
    # Map Kaggle classes to our classes
    class_mapping = {
        'LYMPHOCYTE': 'WBC',
        'MONOCYTE': 'WBC', 
        'EOSINOPHIL': 'WBC',
        'NEUTROPHIL': 'WBC'
        # Note: You'll need to find RBC and Platelet images separately
    }
    
    for split in ['TRAIN', 'TEST']:
        source_split = os.path.join(source_path, 'images', split)
        target_split = 'train' if split == 'TRAIN' else 'test'
        
        for kaggle_class, our_class in class_mapping.items():
            source_class_path = os.path.join(source_split, kaggle_class)
            target_class_path = os.path.join(target_path, target_split, our_class)
            
            if os.path.exists(source_class_path):
                print(f"Copying {kaggle_class} -> {our_class}")
                
                for img_file in os.listdir(source_class_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy2(
                            os.path.join(source_class_path, img_file),
                            os.path.join(target_class_path, f"{kaggle_class}_{img_file}")
                        )

def main():
    print("Dataset Preparation Tool")
    print("=" * 40)
    
    # Get source path from user
    source_path = input("Enter path to downloaded dataset: ").strip()
    
    if not os.path.exists(source_path):
        print(f"Error: Path {source_path} does not exist")
        return
    
    organize_kaggle_dataset(source_path)
    print("Dataset organization complete!")
    
    # Print summary
    for split in ['train', 'test']:
        for cell_type in ['RBC', 'WBC', 'Platelet']:
            path = f'data/real_dataset/{split}/{cell_type}'
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"{split}/{cell_type}: {count} images")

if __name__ == "__main__":
    main()