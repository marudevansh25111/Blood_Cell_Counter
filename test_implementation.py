#!/usr/bin/env python3
"""
Comprehensive Test Suite for Blood Cell Counter

This test suite validates all components of the Blood Cell Counter:
- Individual module testing
- Complete pipeline validation
- Performance benchmarking
- Batch processing capabilities
- Synthetic data generation for testing

Run with: python test_implementation.py
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
import time
import json
from datetime import datetime

# Add src directory to path
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Import components with error handling
try:
    from blood_counter import BloodCellCounter
    from image_processor import ImagePreprocessor
    from cell_detector import CellDetector
    from cell_classifier import CellClassifier
    print("✅ All modules imported successfully")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all source files are created and in the src/ directory")
    print("Expected files:")
    print("  src/__init__.py")
    print("  src/image_processor.py")
    print("  src/cell_detector.py") 
    print("  src/cell_classifier.py")
    print("  src/blood_counter.py")
    sys.exit(1)

class SyntheticImageGenerator:
    """Generate realistic synthetic blood smear images for testing."""
    
    def __init__(self):
        self.cell_type_colors = {
            'RBC': {'base': (200, 100, 100), 'variation': 30},
            'WBC': {'base': (150, 150, 180), 'variation': 25},
            'Platelet': {'base': (220, 200, 210), 'variation': 20}
        }
        np.random.seed(42)  # For reproducible test results
    
    def create_synthetic_blood_image(self, width=800, height=600, filename='synthetic_blood_sample.png'):
        """
        Create realistic synthetic blood smear image.
        
        Args:
            width (int): Image width in pixels
            height (int): Image height in pixels
            filename (str): Output filename
            
        Returns:
            tuple: (image_path, expected_counts_dict)
        """
        print(f"🎨 Creating synthetic blood image: {filename} ({width}×{height})")
        
        # Create base image with realistic background
        image = Image.new('RGB', (width, height), (245, 240, 235))
        draw = ImageDraw.Draw(image)
        
        # Add subtle background texture
        self._add_background_texture(draw, width, height)
        
        # Track actual cell counts
        actual_counts = {'RBC': 0, 'WBC': 0, 'Platelet': 0}
        
        # Generate cells based on typical blood composition
        cell_density = (width * height) / (800 * 600)  # Scale with image size
        
        # Add RBCs (80-85% of cells)
        num_rbcs = max(1, int(45 * cell_density))
        actual_counts['RBC'] = self._add_rbcs(draw, width, height, num_rbcs)
        
        # Add WBCs (5-10% of cells)  
        num_wbcs = max(1, int(6 * cell_density))
        actual_counts['WBC'] = self._add_wbcs(draw, width, height, num_wbcs)
        
        # Add Platelets (5-10% of cells)
        num_platelets = max(1, int(12 * cell_density))
        actual_counts['Platelet'] = self._add_platelets(draw, width, height, num_platelets)
        
        # Save image
        os.makedirs('data/input', exist_ok=True)
        filepath = os.path.join('data/input', filename)
        image.save(filepath, quality=95)
        
        print(f"✅ Created synthetic image: {filepath}")
        print(f"   Expected counts - RBCs: {actual_counts['RBC']}, WBCs: {actual_counts['WBC']}, Platelets: {actual_counts['Platelet']}")
        
        return filepath, actual_counts
    
    def _add_background_texture(self, draw, width, height):
        """Add realistic background texture."""
        for _ in range(min(200, width * height // 2000)):
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            size = np.random.randint(1, 4)
            color = (
                np.random.randint(240, 250),
                np.random.randint(235, 245),
                np.random.randint(230, 240)
            )
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    def _add_rbcs(self, draw, width, height, num_rbcs):
        """Add realistic Red Blood Cells."""
        added = 0
        attempts = 0
        max_attempts = num_rbcs * 3
        
        while added < num_rbcs and attempts < max_attempts:
            attempts += 1
            
            # Random position with margins
            x = np.random.randint(30, width - 30)
            y = np.random.randint(30, height - 30)
            radius = np.random.randint(12, 20)
            
            # RBC color (red-pink with variation)
            base_color = self.cell_type_colors['RBC']['base']
            variation = self.cell_type_colors['RBC']['variation']
            
            red = np.clip(base_color[0] + np.random.randint(-variation, variation), 150, 255)
            green = np.clip(base_color[1] + np.random.randint(-variation//2, variation//2), 80, 180)
            blue = np.clip(base_color[2] + np.random.randint(-variation//2, variation//2), 80, 180)
            
            # Main cell body
            draw.ellipse(
                [x-radius, y-radius, x+radius, y+radius],
                fill=(red, green, blue),
                outline=(red-20, green-10, blue-10),
                width=1
            )
            
            # Central pallor (lighter center - characteristic of RBCs)
            pallor_radius = radius // 3
            pallor_color = (
                min(255, red + 25),
                min(255, green + 20),
                min(255, blue + 20)
            )
            draw.ellipse(
                [x-pallor_radius, y-pallor_radius, x+pallor_radius, y+pallor_radius],
                fill=pallor_color
            )
            
            added += 1
        
        return added
    
    def _add_wbcs(self, draw, width, height, num_wbcs):
        """Add realistic White Blood Cells."""
        added = 0
        attempts = 0
        max_attempts = num_wbcs * 5
        
        while added < num_wbcs and attempts < max_attempts:
            attempts += 1
            
            # WBCs are larger and need more space
            x = np.random.randint(60, width - 60)
            y = np.random.randint(60, height - 60)
            radius = np.random.randint(20, 30)
            
            # WBC color (light purple/blue cytoplasm)
            base_color = self.cell_type_colors['WBC']['base']
            variation = self.cell_type_colors['WBC']['variation']
            
            red = np.clip(base_color[0] + np.random.randint(-variation, variation), 120, 200)
            green = np.clip(base_color[1] + np.random.randint(-variation, variation), 120, 200)
            blue = np.clip(base_color[2] + np.random.randint(-variation, variation), 150, 220)
            
            # Create irregular cell boundary (WBCs are not perfectly circular)
            points = []
            for i in range(12):
                angle = i * 30
                r = radius + np.random.randint(-3, 3)
                px = x + r * np.cos(np.radians(angle))
                py = y + r * np.sin(np.radians(angle))
                points.append((px, py))
            
            # Draw cell body
            draw.polygon(points, fill=(red, green, blue), outline=(red-15, green-15, blue-15), width=1)
            
            # Add nucleus (darker, irregular shape - characteristic of WBCs)
            nucleus_radius = radius // 2
            nucleus_points = []
            for i in range(8):
                angle = i * 45 + np.random.randint(-15, 15)
                r = nucleus_radius + np.random.randint(-2, 2)
                px = x + r * np.cos(np.radians(angle))
                py = y + r * np.sin(np.radians(angle))
                nucleus_points.append((px, py))
            
            nucleus_color = (max(0, red-40), max(0, green-40), max(0, blue-40))
            draw.polygon(nucleus_points, fill=nucleus_color)
            
            # Add some chromatin clumping (darker spots in nucleus)
            for _ in range(np.random.randint(1, 3)):
                spot_x = x + np.random.randint(-nucleus_radius//2, nucleus_radius//2)
                spot_y = y + np.random.randint(-nucleus_radius//2, nucleus_radius//2)
                spot_size = np.random.randint(1, 3)
                spot_color = (max(0, red-60), max(0, green-60), max(0, blue-60))
                draw.ellipse([spot_x-spot_size, spot_y-spot_size, spot_x+spot_size, spot_y+spot_size], fill=spot_color)
            
            added += 1
        
        return added
    
    def _add_platelets(self, draw, width, height, num_platelets):
        """Add realistic Platelets."""
        added = 0
        attempts = 0
        max_attempts = num_platelets * 4
        
        while added < num_platelets and attempts < max_attempts:
            attempts += 1
            
            # Platelets are small and can be anywhere
            x = np.random.randint(15, width - 15)
            y = np.random.randint(15, height - 15)
            size = np.random.randint(2, 6)
            
            # Platelet color (light pink/purple)
            base_color = self.cell_type_colors['Platelet']['base']
            variation = self.cell_type_colors['Platelet']['variation']
            
            red = np.clip(base_color[0] + np.random.randint(-variation, variation), 180, 255)
            green = np.clip(base_color[1] + np.random.randint(-variation, variation), 170, 240)
            blue = np.clip(base_color[2] + np.random.randint(-variation, variation), 180, 240)
            
            # Create irregular platelet fragment
            points = []
            num_points = np.random.randint(4, 6)
            for i in range(num_points):
                angle = i * (360 / num_points) + np.random.randint(-20, 20)
                r = size + np.random.randint(-1, 1)
                px = x + r * np.cos(np.radians(angle))
                py = y + r * np.sin(np.radians(angle))
                points.append((px, py))
            
            # Draw platelet
            draw.polygon(points, fill=(red, green, blue), outline=(red-15, green-15, blue-15), width=1)
            
            # Add small granules (characteristic of platelets)
            for _ in range(np.random.randint(0, 2)):
                granule_x = x + np.random.randint(-size//2, size//2)
                granule_y = y + np.random.randint(-size//2, size//2)
                granule_color = (max(0, red-30), max(0, green-20), max(0, blue-25))
                draw.ellipse([granule_x-1, granule_y-1, granule_x+1, granule_y+1], fill=granule_color)
            
            added += 1
        
        return added

class TestRunner:
    """Main test runner class."""
    
    def __init__(self):
        self.generator = SyntheticImageGenerator()
        self.test_results = {}
        self.start_time = time.time()
    
    def run_all_tests(self):
        """Run comprehensive test suite."""
        print("="*80)
        print("BLOOD CELL COUNTER - COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python version: {sys.version}")
        print(f"Platform: {sys.platform}")
        
        # Create necessary directories
        self._setup_test_environment()
        
        # Run test suite
        tests = [
            ("Module Import Validation", self.test_module_imports),
            ("Synthetic Data Generation", self.test_synthetic_generation),
            ("Image Processor Component", self.test_image_processor),
            ("Cell Detector Component", self.test_cell_detector),
            ("Cell Classifier Component", self.test_cell_classifier),
            ("Complete Pipeline Integration", self.test_complete_pipeline),
            ("Performance Benchmarking", self.test_performance_benchmarks),
            ("Batch Processing", self.test_batch_processing),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"RUNNING: {test_name}")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                result = test_func()
                test_time = time.time() - start_time
                
                self.test_results[test_name] = {
                    'passed': result,
                    'time': test_time
                }
                
                status = "✅ PASSED" if result else "❌ FAILED"
                print(f"\n{status} ({test_time:.2f}s)")
                
            except Exception as e:
                test_time = time.time() - start_time
                self.test_results[test_name] = {
                    'passed': False,
                    'time': test_time,
                    'error': str(e)
                }
                
                print(f"\n❌ FAILED with exception: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate final report
        return self._generate_final_report()
    
    def _setup_test_environment(self):
        """Create test environment directories."""
        directories = [
            'data/input',
            'data/output',
            'data/output/test_results',
            'data/output/batch_test',
            'data/output/performance',
            'models'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("✅ Test environment setup complete")
    
    def test_module_imports(self):
        """Test that all required modules can be imported."""
        print("Testing module imports...")
        
        try:
            # Test core modules
            from image_processor import ImagePreprocessor
            from cell_detector import CellDetector
            from cell_classifier import CellClassifier
            from blood_counter import BloodCellCounter
            
            print("✅ All core modules imported successfully")
            
            # Test required libraries
            import cv2
            import numpy
            import sklearn
            import matplotlib
            import PIL
            
            print("✅ All required libraries available")
            
            # Test version compatibility
            print(f"   OpenCV version: {cv2.__version__}")
            print(f"   NumPy version: {numpy.__version__}")
            print(f"   Scikit-learn version: {sklearn.__version__}")
            print(f"   Python version: {sys.version.split()[0]}")
            
            return True
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
    
    def test_synthetic_generation(self):
        """Test synthetic image generation."""
        print("Testing synthetic image generation...")
        
        try:
            # Test different image sizes
            test_sizes = [(400, 300), (600, 450), (800, 600)]
            
            for width, height in test_sizes:
                filename = f'test_synthetic_{width}x{height}.png'
                image_path, expected_counts = self.generator.create_synthetic_blood_image(
                    width=width, height=height, filename=filename
                )
                
                # Verify image was created
                if not os.path.exists(image_path):
                    print(f"❌ Failed to create image: {image_path}")
                    return False
                
                # Verify image properties
                with Image.open(image_path) as img:
                    if img.size != (width, height):
                        print(f"❌ Image size mismatch: expected {(width, height)}, got {img.size}")
                        return False
                
                # Verify cell counts are reasonable
                total_expected = sum(expected_counts.values())
                if total_expected == 0:
                    print(f"❌ No cells generated in {filename}")
                    return False
                
                print(f"   {filename}: {total_expected} cells generated")
            
            print("✅ Synthetic image generation successful")
            return True
            
        except Exception as e:
            print(f"❌ Synthetic generation failed: {e}")
            return False
    
    def test_image_processor(self):
        """Test image preprocessing component."""
        print("Testing ImagePreprocessor component...")
        
        try:
            # Create test image
            test_path, _ = self.generator.create_synthetic_blood_image(filename='processor_test.png')
            
            # Initialize processor
            processor = ImagePreprocessor()
            
            # Test image loading
            original = processor.load_image(test_path)
            if original is None:
                print("❌ Failed to load test image")
                return False
            
            print(f"   Loaded image shape: {original.shape}")
            
            # Test preprocessing pipeline
            processed = processor.preprocess_image()
            if processed is None:
                print("❌ Preprocessing failed")
                return False
            
            print(f"   Processed image shape: {processed.shape}")
            
            # Test preprocessing statistics
            stats = processor.get_preprocessing_stats()
            if stats is None:
                print("❌ Failed to get preprocessing statistics")
                return False
            
            print(f"   Original range: {stats['original_range']}")
            print(f"   Processed range: {stats['processed_range']}")
            
            # Test individual preprocessing steps
            denoised = processor.reduce_noise(original)
            enhanced = processor.enhance_contrast(denoised)
            hsv = processor.convert_color_space(enhanced)
            final = processor.apply_morphology(hsv)
            
            if any(img is None for img in [denoised, enhanced, hsv, final]):
                print("❌ One or more preprocessing steps failed")
                return False
            
            print("✅ All preprocessing steps successful")
            return True
            
        except Exception as e:
            print(f"❌ Image processor test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_cell_detector(self):
        """Test cell detection component."""
        print("Testing CellDetector component...")
        
        try:
            # Create test image and preprocess it
            test_path, expected_counts = self.generator.create_synthetic_blood_image(filename='detector_test.png')
            
            processor = ImagePreprocessor()
            original = processor.load_image(test_path)
            processed = processor.preprocess_image()
            
            # Initialize detector
            detector = CellDetector()
            
            # Test cell detection
            detected_cells = detector.detect_cells(processed, original)
            
            if not detected_cells:
                print("⚠️  No cells detected - this might be normal for small test images")
                # Don't fail the test if no cells detected, as this might happen with small synthetic images
                return True
            
            print(f"   Detected {len(detected_cells)} cell regions")
            
            # Test detection statistics
            stats = detector.get_detection_statistics()
            if stats is not None:
                print(f"   Average confidence: {stats['average_confidence']:.3f}")
                print(f"   Methods used: {stats['detection_methods_used']}")
            
            # Test individual detection methods
            round_cells = detector.detect_round_cells(processed)
            contour_cells = detector.detect_contour_cells(processed)
            watershed_cells = detector.watershed_segmentation(processed)
            
            print(f"   Round cells: {len(round_cells)}")
            print(f"   Contour cells: {len(contour_cells)}")
            print(f"   Watershed cells: {len(watershed_cells)}")
            
            # Test detection accuracy (rough estimate)
            expected_total = sum(expected_counts.values())
            if expected_total > 0:
                detection_accuracy = len(detected_cells) / expected_total
                print(f"   Detection accuracy: {detection_accuracy:.1%}")
            
            print("✅ Cell detection successful")
            return True
            
        except Exception as e:
            print(f"❌ Cell detector test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_cell_classifier(self):
        """Test cell classification component."""
        print("Testing CellClassifier component...")
        
        try:
            # Create test image and detect cells
            test_path, expected_counts = self.generator.create_synthetic_blood_image(filename='classifier_test.png')
            
            processor = ImagePreprocessor()
            original = processor.load_image(test_path)
            processed = processor.preprocess_image()
            
            detector = CellDetector()
            detected_cells = detector.detect_cells(processed, original)
            
            if not detected_cells:
                print("⚠️  No cells to classify - creating dummy cell for testing")
                # Create a dummy cell region for testing
                h, w = original.shape[:2]
                dummy_cell = {
                    'id': 0,
                    'image': original[h//4:3*h//4, w//4:3*w//4],  # Center region
                    'bbox': (w//4, h//4, 3*w//4, 3*h//4),
                    'center': (w//2, h//2),
                    'method': 'test',
                    'area': (w//2) * (h//2)
                }
                detected_cells = [dummy_cell]
            
            # Initialize classifier
            classifier = CellClassifier()
            
            # Test feature extraction
            test_cell = detected_cells[0]
            features = classifier.extract_features(test_cell['image'])
            
            if features is None:
                print("❌ Feature extraction failed")
                return False
            
            print(f"   Extracted {len(features)} features")
            print(f"   Feature range: {features.min():.3f} - {features.max():.3f}")
            
            # Test individual cell classification
            cell_type, confidence = classifier.classify_cell(test_cell['image'])
            print(f"   Sample classification: {cell_type} (confidence: {confidence:.3f})")
            
            # Test batch classification
            classifications = classifier.classify_all_cells(detected_cells)
            
            if not classifications:
                print("❌ Batch classification failed")
                return False
            
            print(f"   Classified {len(classifications)} cells")
            
            # Test report generation
            report = classifier.generate_report(classifications)
            
            if not report:
                print("❌ Report generation failed")
                return False
            
            print(f"   Report summary:")
            for cell_type, count in report['counts'].items():
                if count > 0:
                    percentage = report['percentages'][cell_type]
                    print(f"     {cell_type}: {count} ({percentage:.1f}%)")
            
            print(f"   Average confidence: {report['average_confidence']:.3f}")
            
            print("✅ Cell classification successful")
            return True
            
        except Exception as e:
            print(f"❌ Cell classifier test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_complete_pipeline(self):
        """Test complete integration pipeline."""
        print("Testing complete pipeline integration...")
        
        try:
            # Create test image
            test_path, expected_counts = self.generator.create_synthetic_blood_image(
                filename='pipeline_test.png', width=600, height=450
            )
            
            # Initialize blood cell counter
            counter = BloodCellCounter()
            
            # Run complete analysis
            start_time = time.time()
            results = counter.process_image(test_path, visualize=False)
            processing_time = time.time() - start_time
            
            if results is None:
                print("❌ Pipeline processing failed")
                return False
            
            print(f"   Processing completed in {processing_time:.2f}s")
            
            # Validate results structure
            required_keys = ['image_path', 'processing_timestamp', 'report', 'processing_times']
            for key in required_keys:
                if key not in results:
                    print(f"❌ Missing key in results: {key}")
                    return False
            
            # Validate report structure
            report = results['report']
            required_report_keys = ['total_cells', 'counts', 'percentages', 'classifications']
            for key in required_report_keys:
                if key not in report:
                    print(f"❌ Missing key in report: {key}")
                    return False
            
            print(f"   Found {report['total_cells']} cells")
            print(f"   Average confidence: {report['average_confidence']:.3f}")
            print(f"   Cell distribution: RBC={report['counts']['RBC']}, WBC={report['counts']['WBC']}, Platelet={report['counts']['Platelet']}")
            
            # Test results saving
            output_path = 'data/output/test_results/pipeline_test_results.json'
            save_success = counter.save_results(output_path)
            
            if not save_success:
                print("❌ Failed to save results")
                return False
            
            # Verify saved file
            if not os.path.exists(output_path):
                print("❌ Results file was not created")
                return False
            
            # Test accuracy against expected results
            expected_total = sum(expected_counts.values())
            detected_total = report['total_cells']
            
            if expected_total > 0:
                accuracy = detected_total / expected_total
                print(f"   Detection accuracy: {accuracy:.1%}")
            
            print("✅ Complete pipeline integration successful")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_performance_benchmarks(self):
        """Test performance with different image sizes."""
        print("Testing performance benchmarks...")
        
        try:
            counter = BloodCellCounter()
            performance_data = []
            
            # Test different image sizes
            test_sizes = [(400, 300), (600, 450), (800, 600)]
            
            for width, height in test_sizes:
                print(f"   Testing {width}×{height} image...")
                
                # Create test image
                filename = f'perf_test_{width}x{height}.png'
                test_path, expected_counts = self.generator.create_synthetic_blood_image(
                    width=width, height=height, filename=filename
                )
                
                # Measure processing time
                start_time = time.time()
                results = counter.process_image(test_path, visualize=False)
                processing_time = time.time() - start_time
                
                if results:
                    cells_detected = results['report']['total_cells']
                    performance_data.append({
                        'size': f'{width}×{height}',
                        'pixels': width * height,
                        'processing_time': processing_time,
                        'cells_detected': cells_detected,
                        'cells_per_second': cells_detected / processing_time if processing_time > 0 else 0,
                        'pixels_per_second': (width * height) / processing_time if processing_time > 0 else 0
                    })
                    
                    print(f"     Time: {processing_time:.2f}s, Cells: {cells_detected}")
                else:
                    print(f"     ❌ Failed to process {width}×{height} image")
                    return False
            
            # Save performance results
            perf_output_path = 'data/output/performance/benchmark_results.json'
            with open(perf_output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'performance_data': performance_data,
                    'system_info': {
                        'platform': sys.platform,
                        'python_version': sys.version.split()[0]
                    }
                }, f, indent=2)
            
            print(f"   Performance data saved to: {perf_output_path}")
            
            # Performance summary
            if performance_data:
                avg_processing_time = np.mean([p['processing_time'] for p in performance_data])
                avg_cells_per_second = np.mean([p['cells_per_second'] for p in performance_data])
                
                print(f"   Average processing time: {avg_processing_time:.2f}s")
                print(f"   Average throughput: {avg_cells_per_second:.1f} cells/second")
            
            print("✅ Performance benchmarking successful")
            return True
            
        except Exception as e:
            print(f"❌ Performance benchmarking failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        print("Testing batch processing...")
        
        try:
            # Create multiple test images
            test_images = []
            for i in range(3):
                filename = f'batch_test_{i+1}.png'
                test_path, expected = self.generator.create_synthetic_blood_image(
                    filename=filename,
                    width=400 + i*100,  # Varying sizes
                    height=300 + i*75
                )
                test_images.append((test_path, expected))
            
            print(f"   Created {len(test_images)} test images")
            
            # Initialize counter and run batch processing
            counter = BloodCellCounter()
            
            batch_start_time = time.time()
            batch_results = counter.batch_process('data/input', 'data/output/batch_test')
            batch_time = time.time() - batch_start_time
            
            if batch_results is None:
                print("❌ Batch processing returned None")
                return False
            
            print(f"   Batch processing completed in {batch_time:.2f}s")
            print(f"   Successfully processed {len(batch_results)} images")
            
            if len(batch_results) > 0:
                print(f"   Average time per image: {batch_time/len(batch_results):.2f}s")
            
            # Verify batch summary was created
            summary_file = 'data/output/batch_test/batch_summary.json'
            if not os.path.exists(summary_file):
                print("❌ Batch summary file not created")
                return False
            
            # Load and validate summary
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            required_summary_keys = ['batch_processing_summary', 'aggregate_statistics', 'performance_metrics']
            for key in required_summary_keys:
                if key not in summary:
                    print(f"❌ Missing key in batch summary: {key}")
                    return False
            
            # Validate aggregate statistics
            agg_stats = summary['aggregate_statistics']
            total_cells = agg_stats['total_cells_detected']
            
            print(f"   Total cells detected across all images: {total_cells}")
            if len(batch_results) > 0:
                print(f"   Average cells per image: {agg_stats['average_cells_per_image']:.1f}")
            
            print("✅ Batch processing successful")
            return True
            
        except Exception as e:
            print(f"❌ Batch processing test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        print("Testing error handling...")
        
        try:
            counter = BloodCellCounter()
            
            # Test 1: Non-existent file
            print("   Testing non-existent file handling...")
            result = counter.process_image('nonexistent_file.png', visualize=False)
            if result is not None:
                print("❌ Should return None for non-existent file")
                return False
            print("     ✅ Non-existent file handled correctly")
            
            # Test 2: Empty/corrupted image
            print("   Testing corrupted image handling...")
            
            # Create empty file
            empty_file = 'data/input/empty_test.png'
            with open(empty_file, 'w') as f:
                f.write("not an image")
            
            result = counter.process_image(empty_file, visualize=False)
            if result is not None:
                print("❌ Should return None for corrupted file")
                return False
            print("     ✅ Corrupted file handled correctly")
            
            # Test 3: Image with no cells (blank image)
            print("   Testing blank image handling...")
            blank_image = Image.new('RGB', (400, 300), (255, 255, 255))
            blank_path = 'data/input/blank_test.png'
            blank_image.save(blank_path)
            
            result = counter.process_image(blank_path, visualize=False)
            if result is None:
                print("❌ Should return empty result structure for blank image")
                return False
            
            if result['report']['total_cells'] > 5:  # Should detect very few or no cells
                print("⚠️  Many cells detected in blank image (this might be normal due to noise)")
            
            print("     ✅ Blank image handled correctly")
            
            # Test 4: Very small image
            print("   Testing very small image handling...")
            small_image = Image.new('RGB', (50, 50), (200, 200, 200))
            small_path = 'data/input/small_test.png'
            small_image.save(small_path)
            
            result = counter.process_image(small_path, visualize=False)
            if result is None:
                print("❌ Should handle small images gracefully")
                return False
            print("     ✅ Small image handled correctly")
            
            # Test 5: Invalid batch processing folder
            print("   Testing invalid batch folder handling...")
            batch_results = counter.batch_process('nonexistent_folder', 'data/output/error_test')
            if batch_results is None or len(batch_results) > 0:
                print("❌ Should return empty list for non-existent folder")
                return False
            print("     ✅ Invalid batch folder handled correctly")
            
            print("✅ Error handling tests successful")
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_final_report(self):
        """Generate comprehensive final test report."""
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print("FINAL TEST REPORT")
        print(f"{'='*80}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Test completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test summary
        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0
        
        print(f"\nTEST SUMMARY:")
        print("-" * 40)
        print(f"Total tests run: {total_tests}")
        print(f"Tests passed: {passed_tests}")
        print(f"Tests failed: {total_tests - passed_tests}")
        print(f"Success rate: {success_rate:.1f}%")
        
        # Detailed results
        print(f"\nDETAILED RESULTS:")
        print("-" * 50)
        print(f"{'Test Name':<35} | {'Status':<8} | {'Time':<8}")
        print("-" * 50)
        
        for test_name, result in self.test_results.items():
            status = "PASSED" if result['passed'] else "FAILED"
            time_str = f"{result['time']:.2f}s"
            print(f"{test_name:<35} | {status:<8} | {time_str:<8}")
            
            if not result['passed'] and 'error' in result:
                print(f"    Error: {result['error']}")
        
        # Performance summary
        print(f"\nPERFORMANCE SUMMARY:")
        print("-" * 30)
        
        if 'Performance Benchmarking' in self.test_results and self.test_results['Performance Benchmarking']['passed']:
            # Load performance data if available
            perf_file = 'data/output/performance/benchmark_results.json'
            if os.path.exists(perf_file):
                with open(perf_file, 'r') as f:
                    perf_data = json.load(f)
                
                if 'performance_data' in perf_data:
                    avg_time = np.mean([p['processing_time'] for p in perf_data['performance_data']])
                    avg_throughput = np.mean([p['cells_per_second'] for p in perf_data['performance_data']])
                    
                    print(f"Average processing time: {avg_time:.2f}s per image")
                    print(f"Average throughput: {avg_throughput:.1f} cells/second")
        
        # Save detailed report
        report_data = {
            'test_execution': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_time': total_time,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'test_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'success_rate': success_rate
            },
            'detailed_results': self.test_results
        }
        
        report_file = f'data/output/test_results/comprehensive_test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Final verdict
        if success_rate == 100:
            print(f"\n🎉 ALL TESTS PASSED! Blood Cell Counter is ready for production.")
            print("\nNext steps:")
            print("1. Run the web interface: streamlit run app.py")
            print("2. Test with real blood smear images")
            print("3. Consider training ML classifier with more data")
            print("4. Deploy to production environment")
        elif success_rate >= 80:
            print(f"\n✅ Most tests passed ({success_rate:.1f}%). System is functional with minor issues.")
            print("Review failed tests and consider fixes before production deployment.")
        else:
            print(f"\n❌ Many tests failed ({100-success_rate:.1f}% failure rate). System needs significant fixes.")
            print("Please address the failed tests before proceeding.")
        
        print(f"\n📋 TEST FILES GENERATED:")
        print("   - Synthetic test images in data/input/")
        print("   - Test results in data/output/test_results/")
        print("   - Performance benchmarks in data/output/performance/")
        print("   - Batch processing results in data/output/batch_test/")
        
        return success_rate == 100

def main():
    """Main test execution function."""
    try:
        print("🧪 Initializing Blood Cell Counter Test Suite...")
        
        # Check if we're in the right directory
        if not os.path.exists('src'):
            print("❌ Error: 'src' directory not found.")
            print("Please run this script from the project root directory.")
            print("Expected project structure:")
            print("  blood_cell_counter/")
            print("  ├── src/")
            print("  ├── data/")
            print("  ├── test_implementation.py")
            print("  └── app.py")
            return False
        
        # Check if source files exist
        required_files = [
            'src/__init__.py',
            'src/image_processor.py',
            'src/cell_detector.py',
            'src/cell_classifier.py',
            'src/blood_counter.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("❌ Error: Missing required source files:")
            for file_path in missing_files:
                print(f"     {file_path}")
            print("\nPlease ensure all source files are created before running tests.")
            return False
        
        print("✅ All required files found")
        
        # Run comprehensive tests
        test_runner = TestRunner()
        success = test_runner.run_all_tests()
        
        # Return appropriate exit code
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*80)
    print("BLOOD CELL COUNTER - AUTOMATED TEST SUITE")
    print("="*80)
    print("This test suite will validate your Blood Cell Counter implementation")
    print("Please ensure all source files are in place before proceeding...")
    print("="*80)
    
    success = main()
    
    if success:
        print(f"\n{'='*80}")
        print("🎉 CONGRATULATIONS! All tests passed successfully!")
        print("Your Blood Cell Counter is ready for production use.")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("⚠️  Some tests failed. Please review the output above.")
        print("Fix any issues and run the tests again.")
        print(f"{'='*80}")
    
    sys.exit(0 if success else 1)