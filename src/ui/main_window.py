"""
Complete Main Window - Full Featured Image Cropping Application
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import time
import threading

class SimpleImageProcessor(QThread):
    """Simplified image processor for the complete application"""
    
    progress_updated = pyqtSignal(int, str)
    file_processed = pyqtSignal(str, bool, str)
    processing_complete = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.input_folder = ""
        self.output_folder = ""
        self.aspect_ratio = "3:4"
        self.quality_preset = "medium"
        self.is_cancelled = False
        self.crop_method = "center"
        
    def setup_processing(self, input_folder, output_folder, aspect_ratio="3:4", 
                        quality_preset="medium", crop_method="center"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.aspect_ratio = aspect_ratio
        self.quality_preset = quality_preset
        self.crop_method = crop_method
        self.is_cancelled = False
    
    def cancel_processing(self):
        self.is_cancelled = True
    
    def run(self):
        try:
            # Get image files
            image_files = self.get_image_files()
            
            if not image_files:
                self.processing_complete.emit({'error': 'No image files found'})
                return
            
            os.makedirs(self.output_folder, exist_ok=True)
            
            total_files = len(image_files)
            successful_files = 0
            failed_files = 0
            
            start_time = time.time()
            
            for i, image_file in enumerate(image_files):
                if self.is_cancelled:
                    break
                
                try:
                    # Process single file
                    success = self.process_single_file(image_file)
                    
                    if success:
                        successful_files += 1
                        self.file_processed.emit(image_file.name, True, "Successfully processed")
                    else:
                        failed_files += 1
                        self.file_processed.emit(image_file.name, False, "Processing failed")
                    
                    # Update progress
                    progress = int((i + 1) / total_files * 100)
                    self.progress_updated.emit(progress, image_file.name)
                    
                except Exception as e:
                    failed_files += 1
                    self.file_processed.emit(image_file.name, False, str(e))
            
            # Processing complete
            end_time = time.time()
            duration = end_time - start_time
            
            stats = {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'duration_seconds': duration,
                'success_rate': successful_files / total_files * 100 if total_files > 0 else 0
            }
            
            self.processing_complete.emit(stats)
            
        except Exception as e:
            self.processing_complete.emit({'error': str(e)})
    
    def get_image_files(self):
        """Get list of image files"""
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        input_path = Path(self.input_folder)
        for ext in supported_formats:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        return sorted(list(set(image_files)))
    
    def process_single_file(self, file_path):
        """Process a single image file"""
        try:
            # Load image
            image = cv2.imread(str(file_path))
            if image is None:
                return False
            
            # Get target dimensions
            aspect_ratios = {
                '1:1': (1, 1),
                '3:4': (3, 4),
                '4:3': (4, 3),
                '16:9': (16, 9)
            }
            
            aspect_w, aspect_h = aspect_ratios.get(self.aspect_ratio, (3, 4))
            
            # Crop image
            cropped = self.crop_image(image, aspect_w / aspect_h)
            
            # Save processed image
            output_filename = f"cropped_{file_path.stem}.png"
            output_path = Path(self.output_folder) / output_filename
            
            # Get compression settings
            compression_params = self.get_compression_params()
            
            return cv2.imwrite(str(output_path), cropped, compression_params)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return False
    
    def crop_image(self, image, target_aspect):
        """Crop image to target aspect ratio"""
        h, w = image.shape[:2]
        current_aspect = w / h
        
        if abs(current_aspect - target_aspect) < 0.01:
            return image
        
        # Calculate crop dimensions
        if current_aspect > target_aspect:
            # Image is wider than target - crop width
            new_w = int(h * target_aspect)
            new_h = h
            start_x = (w - new_w) // 2
            start_y = 0
        else:
            # Image is taller than target - crop height
            new_w = w
            new_h = int(w / target_aspect)
            start_x = 0
            start_y = (h - new_h) // 2
        
        # Extract crop
        cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
        return cropped
    
    def get_compression_params(self):
        """Get compression parameters"""
        quality_settings = {
            'low': [cv2.IMWRITE_PNG_COMPRESSION, 9],
            'medium': [cv2.IMWRITE_PNG_COMPRESSION, 6],
            'high': [cv2.IMWRITE_PNG_COMPRESSION, 1]
        }
        return quality_settings.get(self.quality_preset, quality_settings['medium'])

class MainWindow(QMainWindow):
    """Complete main application window with full features"""
    
    def __init__(self, config_manager=None):
        super().__init__()
        self.config_manager = config_manager
        self.processor = SimpleImageProcessor()
        self.processing_active = False
        
        self.setWindowTitle("Image Auto-Cropping & Augmentation Tool v2.0")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self.setup_ui()
        self.setup_connections()
        self.apply_theme()
        
    def setup_ui(self):
        """Setup the complete user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Header
        self.create_header(layout)
        
        # Main content in tabs
        tab_widget = QTabWidget()
        
        # Processing tab
        processing_tab = self.create_processing_tab()
        tab_widget.addTab(processing_tab, "🖼️ Processing")
        
        # Settings tab
        settings_tab = self.create_settings_tab()
        tab_widget.addTab(settings_tab, "⚙️ Settings")
        
        # Preview tab
        preview_tab = self.create_preview_tab()
        tab_widget.addTab(preview_tab, "👁️ Preview")
        
        layout.addWidget(tab_widget)
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self, layout):
        """Create application header"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                    stop:0 #4CAF50, stop:1 #45a049);
                border-radius: 10px;
                margin: 5px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        
        title_label = QLabel("🖼️ Image Auto-Cropping & Augmentation Tool v2.0")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: white; 
            margin: 15px;
            background: transparent;
        """)
        header_layout.addWidget(title_label)
        
        subtitle = QLabel("Professional Image Processing with Face Detection & Auto-Augmentation")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 12px; 
            color: white; 
            margin-bottom: 15px;
            background: transparent;
        """)
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_frame)
    
    def create_processing_tab(self):
        """Create main processing tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Input/Output section
        io_group = QGroupBox("📁 Input & Output")
        io_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        io_layout = QGridLayout(io_group)
        
        # Input folder
        io_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("Select folder containing images...")
        self.input_browse_btn = QPushButton("Browse")
        self.input_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        io_layout.addWidget(self.input_path_edit, 0, 1)
        io_layout.addWidget(self.input_browse_btn, 0, 2)
        
        # Output folder
        io_layout.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("Select output destination...")
        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        io_layout.addWidget(self.output_path_edit, 1, 1)
        io_layout.addWidget(self.output_browse_btn, 1, 2)
        
        # File info
        self.file_info_label = QLabel("No files selected")
        self.file_info_label.setStyleSheet("color: #666; font-style: italic; margin: 10px;")
        io_layout.addWidget(self.file_info_label, 2, 0, 1, 3)
        
        layout.addWidget(io_group)
        
        # Processing controls
        controls_group = QGroupBox("🎯 Processing Controls")
        controls_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        controls_layout = QHBoxLayout(controls_group)
        
        # Start processing button
        self.start_btn = QPushButton("🚀 Start Processing")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        
        self.cancel_btn = QPushButton("⏹️ Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 15px 25px;
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 8px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        controls_layout.addWidget(self.start_btn, 2)
        controls_layout.addWidget(self.cancel_btn, 1)
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        # Progress section
        progress_group = QGroupBox("📊 Progress")
        progress_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #ddd;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to process images")
        self.status_label.setStyleSheet("font-weight: bold; margin: 5px;")
        progress_layout.addWidget(self.status_label)
        
        # Recent files
        self.recent_files_list = QListWidget()
        self.recent_files_list.setMaximumHeight(120)
        progress_layout.addWidget(QLabel("Recent Files:"))
        progress_layout.addWidget(self.recent_files_list)
        
        layout.addWidget(progress_group)
        
        return widget
    
    def create_settings_tab(self):
        """Create settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Basic settings
        basic_group = QGroupBox("🔧 Basic Settings")
        basic_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        basic_layout = QFormLayout(basic_group)
        
        # Aspect ratio
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["1:1 (Square)", "3:4 (Portrait)", "4:3 (Landscape)", "16:9 (Widescreen)"])
        basic_layout.addRow("Aspect Ratio:", self.aspect_ratio_combo)
        
        # Quality
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low (High Compression)", "Medium (Balanced)", "High (Low Compression)"])
        self.quality_combo.setCurrentIndex(1)  # Medium default
        basic_layout.addRow("Quality:", self.quality_combo)
        
        # Crop method
        self.crop_method_combo = QComboBox()
        self.crop_method_combo.addItems(["Center Crop", "Smart Crop (Coming Soon)", "Face Aware (Coming Soon)"])
        basic_layout.addRow("Crop Method:", self.crop_method_combo)
        
        layout.addWidget(basic_group)
        
        # Advanced settings
        advanced_group = QGroupBox("🔬 Advanced Settings")
        advanced_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        advanced_layout = QFormLayout(advanced_group)
        
        # Theme
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light Theme", "Dark Theme"])
        advanced_layout.addRow("Theme:", self.theme_combo)
        
        # Auto-augmentation (placeholder)
        self.auto_augment_cb = QCheckBox("Enable Auto-Augmentation (Coming Soon)")
        self.auto_augment_cb.setEnabled(False)
        advanced_layout.addRow(self.auto_augment_cb)
        
        layout.addWidget(advanced_group)
        
        # Info section
        info_group = QGroupBox("ℹ️ Information")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        info_layout = QVBoxLayout(info_group)
        
        info_text = QLabel("""
<b>Current Version:</b> v2.0 Complete Edition<br>
<b>Features Available:</b><br>
• Basic image cropping with aspect ratio control<br>
• Multiple quality settings<br>
• Batch processing with progress tracking<br>
• Professional UI with theming support<br><br>

<b>Coming Soon:</b><br>
• Face detection and alignment<br>
• Auto-augmentation for orientation correction<br>
• Smart cropping algorithms<br>
• Preview and analysis tools<br>
• Advanced error handling and logging
        """)
        info_text.setWordWrap(True)
        info_text.setStyleSheet("margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        info_layout.addWidget(info_text)
        
        layout.addWidget(info_group)
        layout.addStretch()
        
        return widget
    
    def create_preview_tab(self):
        """Create preview tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Preview controls
        controls_group = QGroupBox("🎨 Preview Controls")
        controls_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        controls_layout = QHBoxLayout(controls_group)
        
        preview_sample_btn = QPushButton("Preview Sample Image")
        preview_sample_btn.clicked.connect(self.preview_sample_image)
        controls_layout.addWidget(preview_sample_btn)
        
        clear_preview_btn = QPushButton("Clear Preview")
        clear_preview_btn.clicked.connect(self.clear_preview)
        controls_layout.addWidget(clear_preview_btn)
        
        controls_layout.addStretch()
        
        layout.addWidget(controls_group)
        
        # Preview area
        preview_group = QGroupBox("👁️ Image Preview")
        preview_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_scroll = QScrollArea()
        self.preview_label = QLabel("No preview available\n\nSelect input folder and click 'Preview Sample Image' to see how images will be processed.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            border: 2px dashed #ccc; 
            background-color: #f9f9f9;
            padding: 50px;
            color: #666;
            font-size: 14px;
        """)
        self.preview_label.setMinimumSize(400, 300)
        
        self.preview_scroll.setWidget(self.preview_label)
        self.preview_scroll.setWidgetResizable(True)
        preview_layout.addWidget(self.preview_scroll)
        
        layout.addWidget(preview_group)
        
        return widget
    
    def create_status_bar(self):
        """Create status bar"""
        self.statusBar().showMessage("Ready to process images")
        
        # Add permanent widgets
        self.files_label = QLabel("Files: 0")
        self.statusBar().addPermanentWidget(self.files_label)
        
        self.time_label = QLabel("Time: 00:00")
        self.statusBar().addPermanentWidget(self.time_label)
        
    def setup_connections(self):
        """Setup signal connections"""
        # Browse buttons
        self.input_browse_btn.clicked.connect(self.browse_input_folder)
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        
        # Processing controls
        self.start_btn.clicked.connect(self.start_processing)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        
        # Settings
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        
        # Input change
        self.input_path_edit.textChanged.connect(self.update_file_info)
        
        # Processor signals
        self.processor.progress_updated.connect(self.on_progress_updated)
        self.processor.file_processed.connect(self.on_file_processed)
        self.processor.processing_complete.connect(self.on_processing_complete)
    
    def browse_input_folder(self):
        """Browse for input folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_path_edit.setText(folder)
    
    def browse_output_folder(self):
        """Browse for output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_path_edit.setText(folder)
    
    def update_file_info(self):
        """Update file information"""
        input_path = self.input_path_edit.text()
        if input_path and os.path.exists(input_path):
            count = self.count_image_files(input_path)
            self.file_info_label.setText(f"Found {count} image files")
            self.files_label.setText(f"Files: {count}")
        else:
            self.file_info_label.setText("No files selected")
            self.files_label.setText("Files: 0")
    
    def count_image_files(self, folder_path):
        """Count image files in folder"""
        try:
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            count = 0
            for ext in supported_formats:
                count += len(list(Path(folder_path).rglob(f"*{ext}")))
                count += len(list(Path(folder_path).rglob(f"*{ext.upper()}")))
            return count
        except:
            return 0
    
    def start_processing(self):
        """Start image processing"""
        input_folder = self.input_path_edit.text()
        output_folder = self.output_path_edit.text()
        
        if not input_folder or not output_folder:
            QMessageBox.warning(self, "Warning", "Please select both input and output folders.")
            return
        
        if not os.path.exists(input_folder):
            QMessageBox.warning(self, "Warning", "Input folder does not exist.")
            return
        
        # Get settings
        aspect_ratio = self.get_aspect_ratio_key()
        quality_preset = self.get_quality_key()
        crop_method = self.get_crop_method_key()
        
        # Setup processor
        self.processor.setup_processing(
            input_folder=input_folder,
            output_folder=output_folder,
            aspect_ratio=aspect_ratio,
            quality_preset=quality_preset,
            crop_method=crop_method
        )
        
        # Update UI
        self.set_processing_state(True)
        
        # Start processing
        self.processor.start()
    
    def cancel_processing(self):
        """Cancel processing"""
        self.processor.cancel_processing()
        self.set_processing_state(False)
        self.status_label.setText("Processing cancelled")
    
    def get_aspect_ratio_key(self):
        """Get aspect ratio key"""
        aspect_map = {
            "1:1 (Square)": "1:1",
            "3:4 (Portrait)": "3:4",
            "4:3 (Landscape)": "4:3",
            "16:9 (Widescreen)": "16:9"
        }
        return aspect_map.get(self.aspect_ratio_combo.currentText(), "3:4")
    
    def get_quality_key(self):
        """Get quality key"""
        quality_map = {
            "Low (High Compression)": "low",
            "Medium (Balanced)": "medium",
            "High (Low Compression)": "high"
        }
        return quality_map.get(self.quality_combo.currentText(), "medium")
    
    def get_crop_method_key(self):
        """Get crop method key"""
        method_map = {
            "Center Crop": "center",
            "Smart Crop (Coming Soon)": "center",
            "Face Aware (Coming Soon)": "center"
        }
        return method_map.get(self.crop_method_combo.currentText(), "center")
    
    def set_processing_state(self, processing):
        """Update UI for processing state"""
        self.processing_active = processing
        self.start_btn.setEnabled(not processing)
        self.cancel_btn.setEnabled(processing)
        
        if processing:
            self.progress_bar.setValue(0)
            self.status_label.setText("Processing started...")
        else:
            self.status_label.setText("Ready to process images")
    
    def on_progress_updated(self, progress, filename):
        """Handle progress update"""
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processing: {filename}")
    
    def on_file_processed(self, filename, success, message):
        """Handle file processed"""
        status_icon = "✅" if success else "❌"
        item = QListWidgetItem(f"{status_icon} {filename}")
        item.setTextColor(QColor('green') if success else QColor('red'))
        self.recent_files_list.insertItem(0, item)
        
        # Keep only recent 10 items
        while self.recent_files_list.count() > 10:
            self.recent_files_list.takeItem(self.recent_files_list.count() - 1)
    
    def on_processing_complete(self, stats):
        """Handle processing complete"""
        self.set_processing_state(False)
        
        if 'error' in stats:
            QMessageBox.critical(self, "Processing Error", f"Processing failed: {stats['error']}")
            return
        
        # Show completion message
        success_rate = stats.get('success_rate', 0)
        duration = int(stats.get('duration_seconds', 0))
        minutes = duration // 60
        seconds = duration % 60
        
        message = f"""Processing Complete!

Processed: {stats['successful_files']}/{stats['total_files']} files
Success Rate: {success_rate:.1f}%
Duration: {minutes}m {seconds}s

Output saved to: {self.output_path_edit.text()}"""
        
        QMessageBox.information(self, "Processing Complete", message)
    
    def preview_sample_image(self):
        """Preview sample image processing"""
        input_folder = self.input_path_edit.text()
        if not input_folder or not os.path.exists(input_folder):
            QMessageBox.warning(self, "Warning", "Please select a valid input folder first.")
            return
        
        try:
            # Find first image file
            supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            sample_file = None
            
            for ext in supported_formats:
                files = list(Path(input_folder).rglob(f"*{ext}"))
                if files:
                    sample_file = files[0]
                    break
            
            if not sample_file:
                QMessageBox.information(self, "Info", "No supported image files found.")
                return
            
            # Load and process sample
            image = cv2.imread(str(sample_file))
            if image is None:
                QMessageBox.warning(self, "Warning", "Could not load sample image.")
                return
            
            # Get aspect ratio
            aspect_ratio_key = self.get_aspect_ratio_key()
            aspect_ratios = {'1:1': (1, 1), '3:4': (3, 4), '4:3': (4, 3), '16:9': (16, 9)}
            aspect_w, aspect_h = aspect_ratios.get(aspect_ratio_key, (3, 4))
            target_aspect = aspect_w / aspect_h
            
            # Crop preview
            cropped = self.processor.crop_image(image, target_aspect)
            
            # Convert to display
            rgb_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale for display
            pixmap = QPixmap.fromImage(qt_image)
            max_size = 400
            if pixmap.width() > max_size or pixmap.height() > max_size:
                pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setText("")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preview failed: {str(e)}")
    
    def clear_preview(self):
        """Clear preview area"""
        self.preview_label.clear()
        self.preview_label.setText("No preview available\n\nSelect input folder and click 'Preview Sample Image' to see how images will be processed.")
    
    def apply_theme(self):
        """Apply selected theme"""
        theme = self.theme_combo.currentText()
        
        if "Dark" in theme:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QGroupBox {
                    font-weight: bold;
                    border: 2px solid #555555;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 10px;
                    background-color: #353535;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #404040;
                    border: 1px solid #555555;
                    padding: 8px;
                    border-radius: 4px;
                    color: #ffffff;
                }
                QPushButton:hover {
                    background-color: #505050;
                }
                QLineEdit, QComboBox {
                    background-color: #404040;
                    border: 1px solid #555555;
                    padding: 5px;
                    border-radius: 3px;
                    color: #ffffff;
                }
                QProgressBar {
                    border: 1px solid #555555;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #404040;
                    color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
                QListWidget {
                    background-color: #404040;
                    border: 1px solid #555555;
                    color: #ffffff;
                }
            """)
        else:
            self.setStyleSheet("")