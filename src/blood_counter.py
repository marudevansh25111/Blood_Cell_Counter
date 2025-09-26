"""
Main Blood Cell Counter Module

This module orchestrates the complete blood cell analysis pipeline:
- Image preprocessing
- Cell detection 
- Cell classification
- Results generation and visualization
- Batch processing capabilities

Main entry point for the Blood Cell Counter system.
"""

from image_processor import ImagePreprocessor
from cell_detector import CellDetector
from cell_classifier import CellClassifier
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import os
from pathlib import Path

class BloodCellCounter:
    """
    Main orchestration class for the Blood Cell Counter system.
    
    This class integrates all components of the blood cell analysis pipeline
    and provides high-level methods for processing single images or batches
    of images.
    """
    
    def __init__(self):
        """Initialize all components of the blood cell counter."""
        self.preprocessor = ImagePreprocessor()
        self.detector = CellDetector()
        self.classifier = CellClassifier()
        self.results = None
        self.processing_history = []
    
    def process_image(self, image_path, use_ml_classification=False, visualize=True, save_intermediate=False):
        """
        Complete blood cell analysis pipeline for a single image.
        
        Args:
            image_path (str): Path to the input blood smear image
            use_ml_classification (bool): Use ML classifier if trained
            visualize (bool): Show visualization plots
            save_intermediate (bool): Save intermediate processing steps
            
        Returns:
            dict: Complete analysis results, or None if processing fails
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING IMAGE: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Step 1: Load and preprocess image
            print("\n🔍 STEP 1: Image Preprocessing")
            print("-" * 40)
            
            original_image = self.preprocessor.load_image(image_path)
            if original_image is None:
                print("❌ Failed to load image")
                return None
            
            processed_image = self.preprocessor.preprocess_image()
            preprocessing_time = time.time() - start_time
            
            print(f"✅ Preprocessing completed in {preprocessing_time:.2f}s")
            
            # Visualize preprocessing if requested
            if visualize:
                self.preprocessor.visualize_preprocessing_steps()
            
            # Step 2: Detect cells
            print("\n🎯 STEP 2: Cell Detection")
            print("-" * 40)
            
            detection_start = time.time()
            cell_regions = self.detector.detect_cells(processed_image, original_image)
            detection_time = time.time() - detection_start
            
            print(f"✅ Detection completed in {detection_time:.2f}s")
            
            if not cell_regions:
                print("⚠️  Warning: No cells detected in image")
                return self._create_empty_result(image_path, original_image.shape)
            
            # Visualize detections if requested
            if visualize:
                self.detector.visualize_detections(original_image)
            
            # Step 3: Classify cells
            print(f"\n🔬 STEP 3: Cell Classification ({'ML' if use_ml_classification else 'Rule-based'})")
            print("-" * 40)
            
            classification_start = time.time()
            classifications = self.classifier.classify_all_cells(cell_regions, use_ml_classification)
            classification_time = time.time() - classification_start
            
            print(f"✅ Classification completed in {classification_time:.2f}s")
            
            # Step 4: Generate comprehensive report
            print("\n📊 STEP 4: Report Generation")
            print("-" * 40)
            
            report = self.classifier.generate_report(classifications)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Create comprehensive results object
            self.results = {
                'image_path': image_path,
                'processing_timestamp': datetime.now().isoformat(),
                'original_image_shape': original_image.shape,
                'processing_times': {
                    'preprocessing': preprocessing_time,
                    'detection': detection_time,
                    'classification': classification_time,
                    'total': total_time
                },
                'detection_stats': self.detector.get_detection_statistics(),
                'report': report,
                'method_used': 'ML' if use_ml_classification else 'Rule-based'
            }
            
            # Add to processing history
            self.processing_history.append({
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'cells_detected': report['total_cells'],
                'processing_time': total_time
            })
            
            print(f"✅ Complete analysis finished in {total_time:.2f}s")
            print(f"📈 Found {report['total_cells']} cells with {report['average_confidence']:.3f} avg confidence")
            
            # Visualize final results if requested
            if visualize:
                self.visualize_results(original_image)
            
            # Save intermediate results if requested
            if save_intermediate:
                self._save_intermediate_results(image_path)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_empty_result(self, image_path, image_shape):
        """Create empty result structure when no cells are detected."""
        return {
            'image_path': image_path,
            'processing_timestamp': datetime.now().isoformat(),
            'original_image_shape': image_shape,
            'processing_times': {'total': 0.0},
            'report': {
                'total_cells': 0,
                'counts': {'RBC': 0, 'WBC': 0, 'Platelet': 0, 'Unknown': 0},
                'percentages': {'RBC': 0.0, 'WBC': 0.0, 'Platelet': 0.0, 'Unknown': 0.0},
                'average_confidence': 0.0,
                'classifications': []
            }
        }
    
    def _save_intermediate_results(self, image_path):
        """Save intermediate processing results for debugging."""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = f"data/output/intermediate/{base_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed image
        if self.preprocessor.processed_image is not None:
            cv2.imwrite(f"{output_dir}/processed.png", self.preprocessor.processed_image)
        
        # Save detections
        self.detector.save_detections(f"{output_dir}/detections.json")
    
    def visualize_results(self, original_image):
        """
        Create comprehensive visualization of analysis results.
        
        Args:
            original_image (numpy.ndarray): Original input image
        """
        if self.results is None:
            print("❌ No results to visualize")
            return
        
        # Create visualization
        viz_image = original_image.copy()
        classifications = self.results['report']['classifications']
        
        # Color map for different cell types
        color_map = {
            'RBC': (220, 20, 20),        # Red
            'WBC': (20, 220, 20),        # Green  
            'Platelet': (20, 20, 220),   # Blue
            'Unknown': (128, 128, 128)   # Gray
        }
        
        # Draw classifications on image
        for classification in classifications:
            bbox = classification['bbox']
            cell_type = classification['type']
            confidence = classification['confidence']
            
            color = color_map.get(cell_type, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Add confidence-based marker
            center = classification['center']
            marker_size = int(5 + confidence * 5)  # Size based on confidence
            cv2.circle(viz_image, center, marker_size, color, -1)
            
            # Add label with type and confidence
            label = f"{cell_type[:3]} {confidence:.2f}"
            label_pos = (bbox[0], bbox[1] - 10)
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                viz_image,
                (label_pos[0] - 2, label_pos[1] - text_height - 4),
                (label_pos[0] + text_width + 2, label_pos[1] + 2),
                color, -1
            )
            
            # Add text
            cv2.putText(
                viz_image, label, label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Create comprehensive results plot
        fig = plt.figure(figsize=(20, 12))
        
        # Main annotated image
        ax1 = plt.subplot(2, 3, (1, 2))
        ax1.imshow(viz_image)
        ax1.set_title(f'Blood Cell Analysis Results\n'
                     f'Total: {self.results["report"]["total_cells"]} cells | '
                     f'Avg Confidence: {self.results["report"]["average_confidence"]:.3f} | '
                     f'Time: {self.results["processing_times"]["total"]:.1f}s',
                     fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Cell type distribution bar chart
        ax2 = plt.subplot(2, 3, 3)
        counts = self.results['report']['counts']
        cell_types = list(counts.keys())
        cell_counts = list(counts.values())
        colors = ['#dc143c', '#228b22', '#4169e1', '#808080']
        
        # Filter out zero counts for cleaner visualization
        non_zero_types = [t for t, c in zip(cell_types, cell_counts) if c > 0]
        non_zero_counts = [c for c in cell_counts if c > 0]
        non_zero_colors = [colors[i] for i, c in enumerate(cell_counts) if c > 0]
        
        if non_zero_counts:
            bars = ax2.bar(non_zero_types, non_zero_counts, color=non_zero_colors, alpha=0.8)
            ax2.set_title('Cell Type Distribution', fontweight='bold')
            ax2.set_ylabel('Count')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars, non_zero_counts):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Confidence distribution histogram
        ax3 = plt.subplot(2, 3, 4)
        confidences = [c['confidence'] for c in classifications]
        if confidences:
            ax3.hist(confidences, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Confidence Score Distribution', fontweight='bold')
            ax3.set_xlabel('Confidence Score')
            ax3.set_ylabel('Number of Cells')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(np.mean(confidences), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidences):.3f}')
            ax3.legend()
        
        # Processing time breakdown
        ax4 = plt.subplot(2, 3, 5)
        times = self.results['processing_times']
        time_labels = ['Preprocessing', 'Detection', 'Classification']
        time_values = [times['preprocessing'], times['detection'], times['classification']]
        time_colors = ['orange', 'purple', 'brown']
        
        wedges, texts, autotexts = ax4.pie(time_values, labels=time_labels, colors=time_colors,
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Processing Time Breakdown\nTotal: {times["total"]:.2f}s', fontweight='bold')
        
        # Summary statistics table
        ax5 = plt.subplot(2, 3, 6)
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create summary data
        summary_data = []
        for cell_type in cell_types:
            count = counts[cell_type]
            percentage = self.results['report']['percentages'][cell_type]
            if count > 0:
                conf_stats = self.results['report']['confidence_by_type'][cell_type]
                summary_data.append([
                    cell_type,
                    f"{count}",
                    f"{percentage:.1f}%", 
                    f"{conf_stats['mean']:.3f}"
                ])
        
        if summary_data:
            table = ax5.table(cellText=summary_data,
                            colLabels=['Type', 'Count', 'Percentage', 'Avg Conf'],
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for (i, j), cell in table.get_celld().items():
                if i == 0:  # Header row
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(color='white')
                else:
                    cell.set_facecolor('#f0f0f0')
        
        ax5.set_title('Classification Summary', fontweight='bold', pad=20)
        
        plt.suptitle(f'Blood Cell Counter Analysis - {os.path.basename(self.results["image_path"])}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def print_detailed_report(self):
        """Print comprehensive analysis report to console."""
        if self.results is None:
            print("❌ No results available")
            return
        
        report = self.results['report']
        
        print("\n" + "="*70)
        print("BLOOD CELL ANALYSIS REPORT")
        print("="*70)
        
        # Basic information
        print(f"Image File: {os.path.basename(self.results['image_path'])}")
        print(f"Analysis Date: {self.results['processing_timestamp'][:19]}")
        print(f"Image Dimensions: {self.results['original_image_shape']}")
        print(f"Classification Method: {self.results['method_used']}")
        
        # Processing times
        times = self.results['processing_times']
        print(f"\nPROCESSING PERFORMANCE:")
        print("-" * 30)
        print(f"Preprocessing:   {times['preprocessing']:6.2f}s")
        print(f"Detection:       {times['detection']:6.2f}s") 
        print(f"Classification:  {times['classification']:6.2f}s")
        print(f"Total Time:      {times['total']:6.2f}s")
        
        # Detection statistics
        if self.results.get('detection_stats'):
            det_stats = self.results['detection_stats']
            print(f"\nDETECTION STATISTICS:")
            print("-" * 30)
            print(f"Total Cells Detected: {det_stats['total_cells_detected']}")
            print(f"Detection Methods Used: {', '.join(det_stats['detection_methods_used'])}")
            print(f"Average Detection Confidence: {det_stats['average_confidence']:.3f}")
            print(f"Detection Density: {det_stats['detection_density']:.2e} cells/pixel")
        
        # Classification results
        print(f"\nCLASSIFICATION RESULTS:")
        print("-" * 30)
        print(f"Total Cells Classified: {report['total_cells']}")
        print(f"Overall Confidence: {report['average_confidence']:.3f}")
        
        print(f"\nCELL TYPE BREAKDOWN:")
        print("-" * 40)
        print(f"{'Type':>10} | {'Count':>5} | {'Percentage':>10} | {'Avg Conf':>8}")
        print("-" * 40)
        
        for cell_type in ['RBC', 'WBC', 'Platelet', 'Unknown']:
            count = report['counts'][cell_type]
            percentage = report['percentages'][cell_type]
            
            if count > 0:
                conf_stats = report['confidence_by_type'][cell_type]
                print(f"{cell_type:>10} | {count:5d} | {percentage:9.1f}% | {conf_stats['mean']:8.3f}")
        
        # Individual cell details (for smaller sets)
        if report['total_cells'] <= 50:
            print(f"\nINDIVIDUAL CELL CLASSIFICATIONS:")
            print("-" * 60)
            print(f"{'ID':>3} | {'Type':>8} | {'Confidence':>10} | {'Area':>6} | {'Method':>10}")
            print("-" * 60)
            
            for classification in report['classifications']:
                print(f"{classification['id']:3d} | "
                      f"{classification['type']:>8} | "
                      f"{classification['confidence']:10.3f} | "
                      f"{classification['area']:6d} | "
                      f"{classification.get('detection_method', 'N/A'):>10}")
        
        # Clinical interpretation (basic)
        self._print_clinical_interpretation(report)
        
        print("\n" + "="*70)
    
    def _print_clinical_interpretation(self, report):
        """Print basic clinical interpretation of results."""
        print(f"\nCLINICAL INTERPRETATION:")
        print("-" * 30)
        
        total_cells = report['total_cells']
        if total_cells == 0:
            print("No cells detected - image may need better preparation")
            return
        
        rbc_pct = report['percentages']['RBC']
        wbc_pct = report['percentages']['WBC'] 
        platelet_pct = report['percentages']['Platelet']
        
        # Normal ranges (approximate)
        normal_rbc_range = (85, 95)
        normal_wbc_range = (3, 12)
        normal_platelet_range = (2, 8)
        
        interpretations = []
        
        # RBC analysis
        if rbc_pct < normal_rbc_range[0]:
            interpretations.append(f"Low RBC percentage ({rbc_pct:.1f}%) - possible anemia")
        elif rbc_pct > normal_rbc_range[1]:
            interpretations.append(f"High RBC percentage ({rbc_pct:.1f}%) - possible polycythemia")
        else:
            interpretations.append(f"RBC percentage appears normal ({rbc_pct:.1f}%)")
        
        # WBC analysis  
        if wbc_pct > normal_wbc_range[1]:
            interpretations.append(f"Elevated WBC percentage ({wbc_pct:.1f}%) - possible infection")
        elif wbc_pct < normal_wbc_range[0]:
            interpretations.append(f"Low WBC percentage ({wbc_pct:.1f}%) - possible immunosuppression")
        
        # Confidence assessment
        avg_conf = report['average_confidence']
        if avg_conf < 0.6:
            interpretations.append("Low classification confidence - results should be verified")
        elif avg_conf > 0.8:
            interpretations.append("High classification confidence - results are reliable")
        
        for interpretation in interpretations:
            print(f"• {interpretation}")
        
        print("\n⚠️  Note: This is automated analysis for research purposes only.")
        print("   Professional medical review is required for clinical decisions.")
    
    def save_results(self, output_path):
        """
        Save analysis results to JSON file.
        
        Args:
            output_path (str): Path where to save the results
        """
        if self.results is None:
            print("❌ No results to save")
            return False
        
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare data for JSON serialization
            results_to_save = self.results.copy()
            
            # Convert numpy arrays to lists
            if 'original_image_shape' in results_to_save:
                results_to_save['original_image_shape'] = list(results_to_save['original_image_shape'])
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            
            print(f"✅ Results saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def batch_process(self, image_folder, output_folder, use_ml=False):
        """
        Process multiple images in batch mode.
        
        Args:
            image_folder (str): Folder containing input images
            output_folder (str): Folder to save results
            use_ml (bool): Use ML classification if available
            
        Returns:
            list: List of processing results for all images
        """
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING")
        print(f"{'='*70}")
        print(f"Input folder: {image_folder}")
        print(f"Output folder: {output_folder}")
        print(f"Classification: {'ML' if use_ml else 'Rule-based'}")
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f"*{ext}"))
            image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"❌ No image files found in {image_folder}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        batch_results = []
        failed_images = []
        batch_start_time = time.time()
        
        for i, image_file in enumerate(image_files):
            print(f"\n{'='*50}")
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            print(f"{'='*50}")
            
            try:
                # Process image without visualization for batch mode
                result = self.process_image(
                    str(image_file), 
                    use_ml_classification=use_ml,
                    visualize=False
                )
                
                if result:
                    # Save individual result
                    output_file = Path(output_folder) / f"{image_file.stem}_results.json"
                    self.save_results(str(output_file))
                    
                    batch_results.append(result)
                    print(f"✅ Success: {result['report']['total_cells']} cells detected")
                else:
                    failed_images.append(str(image_file))
                    print(f"❌ Failed to process {image_file.name}")
                    
            except Exception as e:
                failed_images.append(str(image_file))
                print(f"❌ Error processing {image_file.name}: {e}")
        
        batch_total_time = time.time() - batch_start_time
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(batch_results, batch_total_time, failed_images)
        
        # Save batch summary
        summary_file = Path(output_folder) / "batch_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        # Print batch results
        self._print_batch_summary(batch_summary)
        
        return batch_results
    
    def _generate_batch_summary(self, batch_results, total_time, failed_images):
        """Generate comprehensive batch processing summary."""
        if not batch_results:
            return {
                'total_images_processed': 0,
                'successful_images': 0,
                'failed_images': len(failed_images),
                'failed_image_list': failed_images,
                'processing_timestamp': datetime.now().isoformat(),
                'total_processing_time': total_time
            }
        
        # Calculate aggregate statistics
        total_cells = sum(r['report']['total_cells'] for r in batch_results)
        total_rbcs = sum(r['report']['counts']['RBC'] for r in batch_results)
        total_wbcs = sum(r['report']['counts']['WBC'] for r in batch_results)
        total_platelets = sum(r['report']['counts']['Platelet'] for r in batch_results)
        
        avg_confidence = np.mean([r['report']['average_confidence'] for r in batch_results])
        avg_processing_time = np.mean([r['processing_times']['total'] for r in batch_results])
        
        # Processing time statistics
        processing_times = [r['processing_times']['total'] for r in batch_results]
        
        summary = {
            'batch_processing_summary': {
                'total_images_attempted': len(batch_results) + len(failed_images),
                'successful_images': len(batch_results),
                'failed_images': len(failed_images),
                'failed_image_list': failed_images,
                'success_rate': len(batch_results) / (len(batch_results) + len(failed_images)) * 100
            },
            'aggregate_statistics': {
                'total_cells_detected': total_cells,
                'average_cells_per_image': total_cells / len(batch_results),
                'total_rbcs': total_rbcs,
                'total_wbcs': total_wbcs,
                'total_platelets': total_platelets,
                'overall_cell_distribution': {
                    'RBC': total_rbcs / total_cells * 100 if total_cells > 0 else 0,
                    'WBC': total_wbcs / total_cells * 100 if total_cells > 0 else 0,
                    'Platelet': total_platelets / total_cells * 100 if total_cells > 0 else 0
                }
            },
            'performance_metrics': {
                'total_processing_time': total_time,
                'average_processing_time_per_image': avg_processing_time,
                'processing_rate': len(batch_results) / total_time * 60,  # images per minute
                'average_confidence': avg_confidence,
                'min_processing_time': min(processing_times),
                'max_processing_time': max(processing_times)
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _print_batch_summary(self, summary):
        """Print formatted batch processing summary."""
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        batch_info = summary['batch_processing_summary']
        print(f"Images Attempted: {batch_info['total_images_attempted']}")
        print(f"Successful: {batch_info['successful_images']}")
        print(f"Failed: {batch_info['failed_images']}")
        print(f"Success Rate: {batch_info['success_rate']:.1f}%")
        
        if batch_info['failed_images'] > 0:
            print(f"Failed Images: {', '.join(batch_info['failed_image_list'])}")
        
        stats = summary['aggregate_statistics']
        print(f"\nAGGREGATE RESULTS:")
        print("-" * 30)
        print(f"Total Cells Detected: {stats['total_cells_detected']:,}")
        print(f"Average Cells per Image: {stats['average_cells_per_image']:.1f}")
        print(f"RBCs: {stats['total_rbcs']:,} ({stats['overall_cell_distribution']['RBC']:.1f}%)")
        print(f"WBCs: {stats['total_wbcs']:,} ({stats['overall_cell_distribution']['WBC']:.1f}%)")
        print(f"Platelets: {stats['total_platelets']:,} ({stats['overall_cell_distribution']['Platelet']:.1f}%)")
        
        perf = summary['performance_metrics']
        print(f"\nPERFORMANCE METRICS:")
        print("-" * 30)
        print(f"Total Processing Time: {perf['total_processing_time']:.1f}s")
        print(f"Average Time per Image: {perf['average_processing_time_per_image']:.2f}s")
        print(f"Processing Rate: {perf['processing_rate']:.1f} images/minute")
        print(f"Average Confidence: {perf['average_confidence']:.3f}")
        print(f"Time Range: {perf['min_processing_time']:.2f}s - {perf['max_processing_time']:.2f}s")
        
        print(f"\n✅ Batch processing completed successfully!")
    
    def get_processing_history(self):
        """
        Get history of all processed images in this session.
        
        Returns:
            list: List of processing history entries
        """
        return self.processing_history
    
    def clear_history(self):
        """Clear processing history."""
        self.processing_history = []
        print("✅ Processing history cleared")