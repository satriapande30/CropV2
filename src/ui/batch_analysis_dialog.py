"""
Batch analysis dialog for checking image dataset consistency
"""

import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.image_augmenter import ImageAugmenter
from core.face_detector import FaceDetector

class BatchAnalysisDialog(QDialog):
    """Dialog for analyzing batch consistency and suggesting corrections"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Analysis")
        self.setMinimumSize(600, 500)
        
        # Initialize analyzers
        self.face_detector = FaceDetector()
        self.image_augmenter = ImageAugmenter(self.face_detector)
        
        # Analysis results
        self.analysis_results = None
        self.sample_images = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup batch analysis UI"""
        layout = QVBoxLayout(self)
        
        # Header
        header_label = QLabel("Batch Consistency Analysis")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)
        
        # Analysis controls
        controls_group = QGroupBox("Analysis Settings")
        controls_layout = QFormLayout(controls_group)
        
        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(5, 50)
        self.sample_size_spin.setValue(20)
        self.sample_size_spin.setToolTip("Number of images to analyze (more = more accurate)")
        controls_layout.addRow("Sample Size:", self.sample_size_spin)
        
        self.analyze_btn = QPushButton("Analyze Folder")
        self.analyze_btn.clicked.connect(self.analyze_folder_dialog)
        controls_layout.addRow(self.analyze_btn)
        
        layout.addWidget(controls_group)
        
        # Results area
        self.results_widget = QTabWidget()
        
        # Summary tab
        self.summary_tab = self.create_summary_tab()
        self.results_widget.addTab(self.summary_tab, "Summary")
        
        # Details tab
        self.details_tab = self.create_details_tab()
        self.results_widget.addTab(self.details_tab, "Details")
        
        # Recommendations tab
        self.recommendations_tab = self.create_recommendations_tab()
        self.results_widget.addTab(self.recommendations_tab, "Recommendations")
        
        layout.addWidget(self.results_widget)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Report")
        export_btn.clicked.connect(self.export_report)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_summary_tab(self):
        """Create summary tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Statistics display
        stats_group = QGroupBox("Analysis Summary")
        stats_layout = QFormLayout(stats_group)
        
        self.total_files_label = QLabel("-")
        self.faces_detected_label = QLabel("-")
        self.face_detection_rate_label = QLabel("-")
        self.rotation_issues_label = QLabel("-")
        self.orientation_issues_label = QLabel("-")
        
        stats_layout.addRow("Total Files:", self.total_files_label)
        stats_layout.addRow("Faces Detected:", self.faces_detected_label)
        stats_layout.addRow("Face Detection Rate:", self.face_detection_rate_label)
        stats_layout.addRow("Rotation Issues:", self.rotation_issues_label)
        stats_layout.addRow("Orientation Issues:", self.orientation_issues_label)
        
        layout.addWidget(stats_group)
        
        # Progress bar (for analysis)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("No analysis performed yet")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        return widget
    
    def create_details_tab(self):
        """Create details tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Details table
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(5)
        self.details_table.setHorizontalHeaderLabels([
            "Filename", "Face Detected", "Rotation Angle", "Orientation", "Issues"
        ])
        self.details_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.details_table)
        return widget
    
    def create_recommendations_tab(self):
        """Create recommendations tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Recommendations list
        self.recommendations_list = QListWidget()
        self.recommendations_list.setStyleSheet("""
            QListWidget::item {
                padding: 8px;
                margin: 2px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)
        
        layout.addWidget(QLabel("Recommended Actions:"))
        layout.addWidget(self.recommendations_list)
        
        # Auto-apply recommendations button
        auto_apply_btn = QPushButton("Auto-Apply Recommendations")
        auto_apply_btn.clicked.connect(self.auto_apply_recommendations)
        auto_apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(auto_apply_btn)
        
        return widget
    
    def analyze_folder_dialog(self):
        """Show folder selection dialog and analyze"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Analyze")
        if folder:
            self.analyze_folder(folder)
    
    def analyze_folder(self, folder_path):
        """Analyze folder for consistency issues"""
        self.status_label.setText("Starting analysis...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        try:
            # Get image files
            image_files = self.get_image_files(folder_path)
            
            if not image_files:
                QMessageBox.warning(self, "Warning", "No supported image files found in the selected folder.")
                return
            
            # Limit to sample size
            sample_size = min(self.sample_size_spin.value(), len(image_files))
            sample_files = image_files[:sample_size]
            
            # Load sample images
            self.sample_images = []
            for i, file_path in enumerate(sample_files):
                self.progress_bar.setValue(int(i / len(sample_files) * 50))  # First 50%
                QApplication.processEvents()
                
                try:
                    image = cv2.imread(str(file_path))
                    if image is not None:
                        self.sample_images.append(image)
                except Exception as e:
                    continue
            
            # Perform analysis
            if self.sample_images:
                self.progress_bar.setValue(60)
                QApplication.processEvents()
                
                self.analysis_results = self.image_augmenter.analyze_batch_consistency(self.sample_images)
                self.analysis_results['total_files'] = len(image_files)
                self.analysis_results['sample_files'] = len(sample_files)
                self.analysis_results['folder_path'] = folder_path
                
                self.progress_bar.setValue(100)
                self.update_display()
                self.status_label.setText("Analysis complete")
                
            else:
                QMessageBox.warning(self, "Warning", "Could not load any images for analysis.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            
        finally:
            self.progress_bar.setVisible(False)
    
    def get_image_files(self, folder_path):
        """Get list of image files from folder"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        folder = Path(folder_path)
        for ext in supported_formats:
            image_files.extend(folder.rglob(f"*{ext}"))
            image_files.extend(folder.rglob(f"*{ext.upper()}"))
        
        return sorted(list(set(image_files)))
    
    def update_display(self):
        """Update display with analysis results"""
        if not self.analysis_results:
            return
        
        # Update summary
        results = self.analysis_results
        
        self.total_files_label.setText(f"{results.get('total_files', 0)} ({results.get('sample_files', 0)} analyzed)")
        self.faces_detected_label.setText(str(results.get('faces_detected', 0)))
        
        face_rate = results.get('face_detection_rate', 0)
        rate_color = "green" if face_rate > 70 else "orange" if face_rate > 40 else "red"
        self.face_detection_rate_label.setText(f"<span style='color: {rate_color}'>{face_rate:.1f}%</span>")
        
        self.rotation_issues_label.setText(str(results.get('rotation_issues', 0)))
        self.orientation_issues_label.setText(str(results.get('orientation_issues', 0)))
        
        # Update details table (simplified for now)
        self.details_table.setRowCount(1)
        self.details_table.setItem(0, 0, QTableWidgetItem("Sample Analysis"))
        self.details_table.setItem(0, 1, QTableWidgetItem(f"{results.get('faces_detected', 0)} faces"))
        self.details_table.setItem(0, 2, QTableWidgetItem(f"{results.get('rotation_issues', 0)} rotated"))
        self.details_table.setItem(0, 3, QTableWidgetItem(f"{results.get('orientation_issues', 0)} upside-down"))
        self.details_table.setItem(0, 4, QTableWidgetItem("See recommendations"))
        
        # Update recommendations
        self.recommendations_list.clear()
        recommendations = results.get('recommendations', [])
        
        for rec in recommendations:
            item = QListWidgetItem(rec)
            if "rotation" in rec.lower():
                item.setIcon(self.style().standardIcon(QStyle.SP_ArrowRight))
            elif "detection" in rec.lower():
                item.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
            else:
                item.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation))
            
            self.recommendations_list.addItem(item)
        
        if not recommendations:
            item = QListWidgetItem("✓ No issues detected - your dataset appears consistent!")
            item.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
            self.recommendations_list.addItem(item)
    
    def auto_apply_recommendations(self):
        """Auto-apply recommendations to parent window settings"""
        if not self.analysis_results:
            QMessageBox.information(self, "Info", "Please run analysis first.")
            return
        
        recommendations = self.analysis_results.get('recommendations', [])
        
        if not recommendations:
            QMessageBox.information(self, "Info", "No recommendations to apply - your dataset looks good!")
            return
        
        # Apply recommendations to parent window if available
        parent_window = self.parent()
        applied_changes = []
        
        for rec in recommendations:
            if "auto-rotation" in rec.lower():
                if hasattr(parent_window, 'auto_augment_cb'):
                    parent_window.auto_augment_cb.setChecked(True)
                    applied_changes.append("Enabled auto-augmentation")
            
            elif "smart crop" in rec.lower():
                if hasattr(parent_window, 'crop_method_combo'):
                    index = parent_window.crop_method_combo.findText("Smart Crop")
                    if index >= 0:
                        parent_window.crop_method_combo.setCurrentIndex(index)
                        applied_changes.append("Changed to Smart Crop method")
        
        if applied_changes:
            changes_text = "\n".join(f"• {change}" for change in applied_changes)
            QMessageBox.information(
                self, "Recommendations Applied", 
                f"Applied the following changes:\n\n{changes_text}\n\nYou can now start processing with optimized settings."
            )
        else:
            QMessageBox.information(
                self, "Info", 
                "Recommendations noted but require manual adjustment in the main window."
            )
    
    def export_report(self):
        """Export analysis report"""
        if not self.analysis_results:
            QMessageBox.information(self, "Info", "Please run analysis first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Report", 
            f"batch_analysis_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.txt",
            "Text Files (*.txt);;HTML Files (*.html)"
        )
        
        if file_path:
            try:
                self.write_report(file_path)
                QMessageBox.information(self, "Success", f"Report exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")
    
    def write_report(self, file_path):
        """Write analysis report to file"""
        results = self.analysis_results
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("BATCH ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {QDateTime.currentDateTime().toString()}\n")
            f.write(f"Folder: {results.get('folder_path', 'Unknown')}\n")
            f.write(f"Total Files: {results.get('total_files', 0)}\n")
            f.write(f"Sample Size: {results.get('sample_files', 0)}\n\n")
            
            f.write("ANALYSIS RESULTS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Faces Detected: {results.get('faces_detected', 0)}\n")
            f.write(f"Face Detection Rate: {results.get('face_detection_rate', 0):.1f}%\n")
            f.write(f"Rotation Issues: {results.get('rotation_issues', 0)}\n")
            f.write(f"Orientation Issues: {results.get('orientation_issues', 0)}\n\n")
            
            recommendations = results.get('recommendations', [])
            if recommendations:
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 15 + "\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
            else:
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 15 + "\n")
                f.write("No issues detected - dataset appears consistent!\n")
            
            f.write(f"\n\nEnd of report\n")