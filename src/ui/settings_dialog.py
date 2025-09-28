"""
Advanced settings dialog for fine-tuning processing parameters
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class SettingsDialog(QDialog):
    """Advanced settings configuration dialog"""
    
    def __init__(self, parent=None, config_manager=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.config = config_manager.config if config_manager else {}
        
        self.setWindowTitle("Advanced Settings")
        self.setMinimumSize(500, 600)
        self.setModal(True)
        
        self.setup_ui()
        self.load_current_settings()
    
    def setup_ui(self):
        """Setup settings dialog UI"""
        layout = QVBoxLayout(self)
        
        # Create tabs
        tab_widget = QTabWidget()
        
        # Face detection settings
        tab_widget.addTab(self.create_face_detection_tab(), "Face Detection")
        
        # Processing settings
        tab_widget.addTab(self.create_processing_tab(), "Processing")
        
        # Performance settings
        tab_widget.addTab(self.create_performance_tab(), "Performance")
        
        # Output settings
        tab_widget.addTab(self.create_output_tab(), "Output")
        
        layout.addWidget(tab_widget)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        defaults_btn = QPushButton("Restore Defaults")
        defaults_btn.clicked.connect(self.restore_defaults)
        button_layout.addWidget(defaults_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        button_layout.addWidget(apply_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept_settings)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
    
    def create_face_detection_tab(self):
        """Create face detection settings tab"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Detection confidence
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.setDecimals(1)
        layout.addRow("Detection Confidence:", self.confidence_spin)
        
        # Eye alignment
        self.eye_alignment_cb = QCheckBox("Enable eye alignment")
        layout.addRow(self.eye_alignment_cb)
        
        # Crop padding
        self.crop_padding_spin = QDoubleSpinBox()
        self.crop_padding_spin.setRange(0.0, 0.5)
        self.crop_padding_spin.setSingleStep(0.05)
        self.crop_padding_spin.setDecimals(2)
        self.crop_padding_spin.setSuffix(" (fraction of face size)")
        layout.addRow("Crop Padding:", self.crop_padding_spin)
        
        # Rotation threshold
        self.rotation_threshold_spin = QDoubleSpinBox()
        self.rotation_threshold_spin.setRange(1.0, 30.0)
        self.rotation_threshold_spin.setSingleStep(1.0)
        self.rotation_threshold_spin.setDecimals(1)
        self.rotation_threshold_spin.setSuffix("°")
        layout.addRow("Rotation Threshold:", self.rotation_threshold_spin)
        
        # Orientation detection
        self.orientation_detection_cb = QCheckBox("Enable orientation detection")
        layout.addRow(self.orientation_detection_cb)
        
        return widget
    
    def create_processing_tab(self):
        """Create processing settings tab"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Supported formats
        formats_group = QGroupBox("Supported Input Formats")
        formats_layout = QVBoxLayout(formats_group)
        
        self.format_checkboxes = {}
        formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
        
        for fmt in formats:
            cb = QCheckBox(fmt.upper())
            self.format_checkboxes[fmt] = cb
            formats_layout.addWidget(cb)
        
        layout.addRow(formats_group)
        
        # Output format
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItems(['PNG', 'JPEG', 'WebP'])
        layout.addRow("Output Format:", self.output_format_combo)
        
        # Preserve metadata
        self.preserve_metadata_cb = QCheckBox("Preserve image metadata")
        layout.addRow(self.preserve_metadata_cb)
        
        return widget
    
    def create_performance_tab(self):
        """Create performance settings tab"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Max workers
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 16)
        layout.addRow("Max Worker Threads:", self.max_workers_spin)
        
        # Batch size
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100)
        layout.addRow("Batch Size:", self.batch_size_spin)
        
        # Memory limit
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(100, 8000)
        self.memory_limit_spin.setSuffix(" MB")
        layout.addRow("Memory Limit:", self.memory_limit_spin)
        
        # Auto-detect CPU cores button
        detect_cores_btn = QPushButton("Auto-detect CPU Cores")
        detect_cores_btn.clicked.connect(self.auto_detect_cores)
        layout.addRow(detect_cores_btn)
        
        return widget
    
    def create_output_tab(self):
        """Create output settings tab"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Output naming
        naming_group = QGroupBox("Output File Naming")
        naming_layout = QFormLayout(naming_group)
        
        self.prefix_edit = QLineEdit()
        naming_layout.addRow("Filename Prefix:", self.prefix_edit)
        
        self.suffix_edit = QLineEdit()
        naming_layout.addRow("Filename Suffix:", self.suffix_edit)
        
        layout.addRow(naming_group)
        
        # Quality settings per format
        quality_group = QGroupBox("Quality Settings")
        quality_layout = QFormLayout(quality_group)
        
        # PNG compression
        self.png_compression_spin = QSpinBox()
        self.png_compression_spin.setRange(0, 9)
        self.png_compression_spin.setToolTip("0 = No compression, 9 = Maximum compression")
        quality_layout.addRow("PNG Compression:", self.png_compression_spin)
        
        # JPEG quality
        self.jpeg_quality_spin = QSpinBox()
        self.jpeg_quality_spin.setRange(10, 100)
        self.jpeg_quality_spin.setSuffix("%")
        quality_layout.addRow("JPEG Quality:", self.jpeg_quality_spin)
        
        layout.addRow(quality_group)
        
        return widget
    
    def load_current_settings(self):
        """Load current settings into dialog"""
        # Face detection settings
        self.confidence_spin.setValue(self.config.get('face_detection_confidence', 0.5))
        self.eye_alignment_cb.setChecked(self.config.get('eye_alignment', True))
        self.crop_padding_spin.setValue(self.config.get('crop_padding', 0.1))
        self.rotation_threshold_spin.setValue(self.config.get('rotation_threshold', 5.0))
        self.orientation_detection_cb.setChecked(self.config.get('orientation_detection', True))
        
        # Processing settings
        supported_formats = self.config.get('supported_formats', ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'])
        for fmt, cb in self.format_checkboxes.items():
            cb.setChecked(fmt in supported_formats)
        
        self.output_format_combo.setCurrentText(self.config.get('output_format', 'PNG'))
        self.preserve_metadata_cb.setChecked(self.config.get('preserve_metadata', False))
        
        # Performance settings
        self.max_workers_spin.setValue(self.config.get('max_workers', 4))
        self.batch_size_spin.setValue(self.config.get('batch_size', 10))
        self.memory_limit_spin.setValue(self.config.get('memory_limit_mb', 1000))
        
        # Output settings
        self.prefix_edit.setText(self.config.get('filename_prefix', 'cropped_'))
        self.suffix_edit.setText(self.config.get('filename_suffix', ''))
        self.png_compression_spin.setValue(self.config.get('png_compression', 6))
        self.jpeg_quality_spin.setValue(self.config.get('jpeg_quality', 90))
    
    def collect_settings(self):
        """Collect settings from dialog"""
        settings = {}
        
        # Face detection
        settings['face_detection_confidence'] = self.confidence_spin.value()
        settings['eye_alignment'] = self.eye_alignment_cb.isChecked()
        settings['crop_padding'] = self.crop_padding_spin.value()
        settings['rotation_threshold'] = self.rotation_threshold_spin.value()
        settings['orientation_detection'] = self.orientation_detection_cb.isChecked()
        
        # Processing
        supported_formats = []
        for fmt, cb in self.format_checkboxes.items():
            if cb.isChecked():
                supported_formats.append(fmt)
        settings['supported_formats'] = supported_formats
        settings['output_format'] = self.output_format_combo.currentText()
        settings['preserve_metadata'] = self.preserve_metadata_cb.isChecked()
        
        # Performance
        settings['max_workers'] = self.max_workers_spin.value()
        settings['batch_size'] = self.batch_size_spin.value()
        settings['memory_limit_mb'] = self.memory_limit_spin.value()
        
        # Output
        settings['filename_prefix'] = self.prefix_edit.text()
        settings['filename_suffix'] = self.suffix_edit.text()
        settings['png_compression'] = self.png_compression_spin.value()
        settings['jpeg_quality'] = self.jpeg_quality_spin.value()
        
        return settings
    
    def apply_settings(self):
        """Apply settings without closing dialog"""
        if self.config_manager:
            settings = self.collect_settings()
            self.config_manager.update(settings)
            self.config_manager.save_config()
            
            QMessageBox.information(self, "Settings Applied", "Settings have been applied successfully!")
    
    def accept_settings(self):
        """Apply settings and close dialog"""
        self.apply_settings()
        self.accept()
    
    def restore_defaults(self):
        """Restore default settings"""
        reply = QMessageBox.question(
            self, "Restore Defaults",
            "Are you sure you want to restore default settings?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            if self.config_manager:
                self.config_manager.reset_to_defaults()
                self.config = self.config_manager.config
                self.load_current_settings()
                
                QMessageBox.information(self, "Defaults Restored", "Default settings have been restored!")
    
    def auto_detect_cores(self):
        """Auto-detect number of CPU cores"""
        import os
        cores = os.cpu_count()
        if cores:
            # Use 75% of available cores, minimum 1
            recommended = max(1, int(cores * 0.75))
            self.max_workers_spin.setValue(recommended)
            
            QMessageBox.information(
                self, "CPU Detection", 
                f"Detected {cores} CPU cores.\nRecommended setting: {recommended} workers."
            )