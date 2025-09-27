# 🔬 Automated Blood Cell Counter

An advanced computer vision system for automated detection, counting, and classification of blood cells in microscopic images. Built with Python, OpenCV, and machine learning techniques, optimized for Mac M3 Pro.

##User Interface
<img width="990" height="556" alt="Screenshot 2025-09-27 at 12 44 59 AM" src="https://github.com/user-attachments/assets/eb073890-a79b-46ff-a7bc-433f135abaee" />
<img width="990" height="556" alt="Screenshot 2025-09-27 at 1 00 41 AM" src="https://github.com/user-attachments/assets/b2a5a5fe-0370-4c82-8fdc-7e787c55aa51" />
<img width="990" height="556" alt="Screenshot 2025-09-27 at 1 00 53 AM" src="https://github.com/user-attachments/assets/f19f3a51-3e79-4fde-9031-3d954a74d1b4" />
<img width="990" height="556" alt="Screenshot 2025-09-27 at 1 01 06 AM" src="https://github.com/user-attachments/assets/92b53841-7641-4ff2-ad32-ea7883f75223" />

##Also, you can see this app live here:https://bloodcellcounterbydm.streamlit.app/




## 📋 Project Overview

This system automates the traditionally manual process of blood cell counting, providing:
- **Automated Detection**: Multi-algorithm approach using HoughCircles, Watershed, and Contour detection
- **Cell Classification**: Rule-based and ML classification of RBCs, WBCs, and Platelets  
- **Professional Web Interface**: Streamlit-based GUI for easy image upload and analysis
- **Batch Processing**: Handle multiple images efficiently
- **Comprehensive Reporting**: Detailed analysis with downloadable results


## 🚀 Quick Start

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) - optimized build
- **Python 3.11+** 
- **8GB+ RAM** (16GB recommended)
- **2GB free disk space**

### Installation

```bash
# 1. Clone or create project directory
mkdir blood_cell_counter
cd blood_cell_counter

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create project structure
mkdir -p src data/{input,output} models

# 5. Copy source files (see file guide below)
# Copy all .py files to their respective locations

# 6. Run tests to verify installation
python test_implementation.py

# 7. Launch web application
streamlit run app.py
```

## 📁 Project Structure

```
blood_cell_counter/
├── src/                           # Source code modules
│   ├── __init__.py               # Package initialization
│   ├── image_processor.py        # Image preprocessing pipeline
│   ├── cell_detector.py          # Multi-algorithm cell detection
│   ├── cell_classifier.py        # Rule-based and ML classification
│   └── blood_counter.py          # Main orchestration class
├── data/                         # Data directories
│   ├── input/                    # Input images
│   └── output/                   # Processing results
├── models/                       # Trained ML models
├── app.py                        # Streamlit web application
├── test_implementation.py        # Comprehensive test suite
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── LICENSE                       # License information
```

## 🔧 Usage

### Web Interface (Recommended)

```bash
# Start the web application
streamlit run app.py

# Open browser to: http://localhost:8501
# Upload blood smear image and click "Analyze"
```

- **Operating System**: macOS 11+ (optimized for Apple Silicon)
- **Python**: 3.11 or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space
- **Display**: 1920×1080 minimum for web interface


## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 👨‍💻 Author

**Devansh Maru**
- GitHub: @marudevansh25111

## 📚 References

1. Digital Image Processing in Medical Applications
2. Computer Vision for Biomedical Image Analysis  
3. Machine Learning in Medical Imaging
4. OpenCV Documentation and Tutorials
5. Scikit-learn User Guide

---

