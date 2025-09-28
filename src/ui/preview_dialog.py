"""
Enhanced preview dialog with before/after comparison and processing history
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pathlib import Path

class PreviewDialog(QDialog):
    """Enhanced preview dialog with multiple view modes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Preview")
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)
        
        # Preview history
        self.preview_history = []
        self.current_sample = None
        
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        """Setup preview dialog UI"""
        layout = QVBoxLayout(self)
        
        # Create tabs
        self.tab_widget = QTabWidget()
        
        # Sample preview tab
        self.sample_tab = self.create_sample_tab()
        self.tab_widget.addTab(self.sample_tab, "Sample Preview")
        
        # Processing history tab
        self.history_tab = self.create_history_tab()
        self.tab_widget.addTab(self.history_tab, "Processing History")
        
        # Batch preview tab
        self.batch_tab = self.create_batch_tab()
        self.tab_widget.addTab(self.batch_tab, "Batch Preview")
        
        layout.addWidget(self.tab_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.save_preview_btn = QPushButton("Save Preview")
        self.export_report_btn = QPushButton("Export Report")
        self.refresh_btn = QPushButton("Refresh")
        
        button_layout.addWidget(self.save_preview_btn)
        button_layout.addWidget(self.export_report_btn)
        button_layout.addWidget(self.refresh_btn)
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_sample_tab(self):
        """Create sample preview tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info section
        info_layout = QHBoxLayout()
        self.sample_info_label = QLabel("No sample loaded")
        self.sample_info_label.setStyleSheet("font-weight: bold; padding: 10px;")
        info_layout.addWidget(self.sample_info_label)
        
        info_layout.addStretch()
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom:"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        zoom_layout.addWidget(self.zoom_slider)
        
        self.zoom_label = QLabel("100%")
        zoom_layout.addWidget(self.zoom_label)
        
        info_layout.addLayout(zoom_layout)
        layout.addLayout(info_layout)
        
        # Image comparison area
        comparison_widget = QSplitter(Qt.Horizontal)
        
        # Original image side
        original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout(original_group)
        
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidgetResizable(True)
        self.original_scroll.setAlignment(Qt.AlignCenter)
        
        self.original_label = QLabel("No image loaded")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(300, 200)
        self.original_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        
        self.original_scroll.setWidget(self.original_label)
        original_layout.addWidget(self.original_scroll)
        
        comparison_widget.addWidget(original_group)
        
        # Preview/processed image side
        preview_group = QGroupBox("Preview with Crop Area")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setAlignment(Qt.AlignCenter)
        
        self.preview_label = QLabel("No preview available")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(300, 200)
        self.preview_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        
        self.preview_scroll.setWidget(self.preview_label)
        preview_layout.addWidget(self.preview_scroll)
        
        comparison_widget.addWidget(preview_group)
        
        layout.addWidget(comparison_widget, 1)
        
        # Crop information
        crop_info_group = QGroupBox("Crop Information")
        crop_info_layout = QFormLayout(crop_info_group)
        
        self.crop_x_label = QLabel("-")
        self.crop_y_label = QLabel("-")
        self.crop_width_label = QLabel("-")
        self.crop_height_label = QLabel("-")
        self.face_detected_label = QLabel("-")
        
        crop_info_layout.addRow("Crop X:", self.crop_x_label)
        crop_info_layout.addRow("Crop Y:", self.crop_y_label)
        crop_info_layout.addRow("Crop Width:", self.crop_width_label)
        crop_info_layout.addRow("Crop Height:", self.crop_height_label)
        crop_info_layout.addRow("Face Detected:", self.face_detected_label)
        
        layout.addWidget(crop_info_group)
        
        return widget
    
    def create_history_tab(self):
        """Create processing history tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_history_btn = QPushButton("Clear History")
        self.save_history_btn = QPushButton("Save History")
        
        controls_layout.addWidget(self.clear_history_btn)
        controls_layout.addWidget(self.save_history_btn)
        controls_layout.addStretch()
        
        self.history_count_label = QLabel("0 items in history")
        controls_layout.addWidget(self.history_count_label)
        
        layout.addLayout(controls_layout)
        
        # History scroll area
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.addStretch()  # Add stretch at end
        
        self.history_scroll.setWidget(self.history_widget)
        layout.addWidget(self.history_scroll)
        
        return widget
    
    def create_batch_tab(self):
        """Create batch preview tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 50)
        self.batch_size_spin.setValue(9)  # 3x3 grid
        
        controls_layout.addWidget(QLabel("Grid Size:"))
        controls_layout.addWidget(self.batch_size_spin)
        
        self.generate_batch_btn = QPushButton("Generate Batch Preview")
        controls_layout.addWidget(self.generate_batch_btn)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Batch preview area
        self.batch_scroll = QScrollArea()
        self.batch_scroll.setWidgetResizable(True)
        
        self.batch_widget = QWidget()
        self.batch_layout = QGridLayout(self.batch_widget)
        
        # Add placeholder
        placeholder = QLabel("Click 'Generate Batch Preview' to see multiple images")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-style: italic; padding: 50px;")
        self.batch_layout.addWidget(placeholder, 0, 0)
        
        self.batch_scroll.setWidget(self.batch_widget)
        layout.addWidget(self.batch_scroll)
        
        return widget
    
    def setup_connections(self):
        """Setup signal connections"""
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        self.clear_history_btn.clicked.connect(self.clear_history)
        self.save_history_btn.clicked.connect(self.save_history)
        self.save_preview_btn.clicked.connect(self.save_preview)
        self.export_report_btn.clicked.connect(self.export_report)
        self.refresh_btn.clicked.connect(self.refresh_preview)
        self.generate_batch_btn.clicked.connect(self.generate_batch_preview)
    
    def show_sample_preview(self, file_path, original_image, preview_image, crop_info):
        """Show sample preview with crop information"""
        self.current_sample = {
            'file_path': file_path,
            'original': original_image,
            'preview': preview_image,
            'crop_info': crop_info
        }
        
        # Update sample info
        file_name = Path(file_path).name
        self.sample_info_label.setText(f"Sample: {file_name}")
        
        # Convert and display images
        self.update_sample_display()
        
        # Update crop info
        self.crop_x_label.setText(str(crop_info.get('crop_x', '-')))
        self.crop_y_label.setText(str(crop_info.get('crop_y', '-')))
        self.crop_width_label.setText(str(crop_info.get('crop_width', '-')))
        self.crop_height_label.setText(str(crop_info.get('crop_height', '-')))
        self.face_detected_label.setText("Yes" if crop_info.get('face_detected', False) else "No")
        
        # Switch to sample tab
        self.tab_widget.setCurrentIndex(0)
    
    def add_processing_result(self, filename, original, processed):
        """Add a processing result to history"""
        result = {
            'filename': filename,
            'original': original,
            'processed': processed,
            'timestamp': QDateTime.currentDateTime()
        }
        
        self.preview_history.append(result)
        
        # Add to history display
        self.add_history_item(result)
        
        # Update count
        self.update_history_count()
        
        # Auto-scroll to bottom
        QTimer.singleShot(100, self.scroll_history_to_bottom)
    
    def add_history_item(self, result):
        """Add an item to the history display"""
        # Create history item frame
        item_frame = QFrame()
        item_frame.setFrameStyle(QFrame.Box)
        item_frame.setStyleSheet("QFrame { border: 1px solid #ddd; margin: 2px; padding: 5px; }")
        item_layout = QHBoxLayout(item_frame)
        
        # Filename and timestamp
        info_layout = QVBoxLayout()
        
        filename_label = QLabel(result['filename'])
        filename_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(filename_label)
        
        time_label = QLabel(result['timestamp'].toString("hh:mm:ss"))
        time_label.setStyleSheet("color: #666; font-size: 10px;")
        info_layout.addWidget(time_label)
        
        item_layout.addLayout(info_layout, 1)
        
        # Original thumbnail
        original_thumb = self.create_thumbnail(result['original'], 80)
        original_label = QLabel()
        original_label.setPixmap(original_thumb)
        original_label.setAlignment(Qt.AlignCenter)
        item_layout.addWidget(original_label)
        
        # Arrow
        arrow_label = QLabel("→")
        arrow_label.setAlignment(Qt.AlignCenter)
        arrow_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        item_layout.addWidget(arrow_label)
        
        # Processed thumbnail
        processed_thumb = self.create_thumbnail(result['processed'], 80)
        processed_label = QLabel()
        processed_label.setPixmap(processed_thumb)
        processed_label.setAlignment(Qt.AlignCenter)
        item_layout.addWidget(processed_label)
        
        # Insert at top (before stretch)
        self.history_layout.insertWidget(0, item_frame)
    
    def create_thumbnail(self, cv_image, size=150):
        """Create a thumbnail QPixmap from OpenCV image"""
        if cv_image is None:
            # Create placeholder
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.lightGray)
            return pixmap
        
        h, w = cv_image.shape[:2]
        
        # Calculate scaling to fit in square
        scale = size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(cv_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        qt_image = QImage(rgb_image.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        
        # Convert to QPixmap and center in square
        pixmap = QPixmap.fromImage(qt_image)
        
        # Create square canvas
        square_pixmap = QPixmap(size, size)
        square_pixmap.fill(Qt.white)
        
        # Draw centered
        painter = QPainter(square_pixmap)
        x = (size - new_w) // 2
        y = (size - new_h) // 2
        painter.drawPixmap(x, y, pixmap)
        painter.end()
        
        return square_pixmap
    
    def on_zoom_changed(self, value):
        """Handle zoom slider change"""
        self.zoom_label.setText(f"{value}%")
        self.update_sample_display()
    
    def update_sample_display(self):
        """Update sample display with current zoom"""
        if not self.current_sample:
            return
        
        zoom_factor = self.zoom_slider.value() / 100.0
        
        # Update original image
        original_pixmap = self.cv_to_pixmap(self.current_sample['original'])
        if zoom_factor != 1.0:
            size = original_pixmap.size() * zoom_factor
            original_pixmap = original_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_label.setPixmap(original_pixmap)
        
        # Update preview image
        preview_pixmap = self.cv_to_pixmap(self.current_sample['preview'])
        if zoom_factor != 1.0:
            size = preview_pixmap.size() * zoom_factor
            preview_pixmap = preview_pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(preview_pixmap)
    
    def cv_to_pixmap(self, cv_image):
        """Convert OpenCV image to QPixmap"""
        if cv_image is None:
            return QPixmap()
        
        h, w, ch = cv_image.shape
        bytes_per_line = ch * w
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qt_image)
    
    def clear_history(self):
        """Clear processing history"""
        reply = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to clear the processing history?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Clear data
            self.preview_history.clear()
            
            # Clear UI
            for i in reversed(range(self.history_layout.count())):
                item = self.history_layout.itemAt(i)
                if item.widget() and item.widget() != self.history_layout.itemAt(-1).widget():  # Don't remove stretch
                    item.widget().deleteLater()
            
            self.update_history_count()
    
    def save_history(self):
        """Save processing history to file"""
        if not self.preview_history:
            QMessageBox.information(self, "Info", "No processing history to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processing History", 
            f"processing_history_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.html",
            "HTML Files (*.html);;Text Files (*.txt)"
        )
        
        if file_path:
            try:
                self.export_history_html(file_path)
                QMessageBox.information(self, "Success", f"History saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save history: {str(e)}")
    
    def export_history_html(self, file_path):
        """Export history as HTML report"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Image Processing History</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { text-align: center; margin-bottom: 30px; }
                .item { border: 1px solid #ddd; margin: 10px 0; padding: 15px; }
                .filename { font-weight: bold; font-size: 16px; }
                .timestamp { color: #666; font-size: 12px; }
                .images { display: flex; align-items: center; margin-top: 10px; }
                .arrow { margin: 0 20px; font-size: 24px; }
                img { max-width: 150px; max-height: 150px; border: 1px solid #ccc; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Image Processing History</h1>
                <p>Generated on: """ + QDateTime.currentDateTime().toString() + """</p>
                <p>Total items: """ + str(len(self.preview_history)) + """</p>
            </div>
        """
        
        for item in reversed(self.preview_history):  # Show newest first
            html_content += f"""
            <div class="item">
                <div class="filename">{item['filename']}</div>
                <div class="timestamp">Processed at: {item['timestamp'].toString()}</div>
                <div class="images">
                    <div>Original</div>
                    <span class="arrow">→</span>
                    <div>Processed</div>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def save_preview(self):
        """Save current preview images"""
        if not self.current_sample:
            QMessageBox.information(self, "Info", "No preview to save.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Preview", 
            f"preview_{Path(self.current_sample['file_path']).stem}.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)"
        )
        
        if file_path:
            try:
                # Create side-by-side comparison
                original = self.current_sample['original']
                preview = self.current_sample['preview']
                
                # Resize to same height
                h1, w1 = original.shape[:2]
                h2, w2 = preview.shape[:2]
                target_height = min(h1, h2, 600)  # Max height 600px
                
                scale1 = target_height / h1
                scale2 = target_height / h2
                
                resized_original = cv2.resize(original, (int(w1 * scale1), target_height))
                resized_preview = cv2.resize(preview, (int(w2 * scale2), target_height))
                
                # Combine horizontally
                combined = np.hstack([resized_original, resized_preview])
                
                # Save
                cv2.imwrite(file_path, combined)
                QMessageBox.information(self, "Success", f"Preview saved to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save preview: {str(e)}")
    
    def export_report(self):
        """Export comprehensive processing report"""
        if not self.preview_history and not self.current_sample:
            QMessageBox.information(self, "Info", "No data to export.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Report", 
            f"processing_report_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.txt",
            "Text Files (*.txt);;CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Image Processing Report\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Generated: {QDateTime.currentDateTime().toString()}\n\n")
                    
                    if self.current_sample:
                        f.write("Current Sample:\n")
                        f.write(f"File: {self.current_sample['file_path']}\n")
                        f.write(f"Crop Info: {self.current_sample['crop_info']}\n\n")
                    
                    if self.preview_history:
                        f.write(f"Processing History ({len(self.preview_history)} items):\n")
                        f.write("-" * 30 + "\n")
                        
                        for i, item in enumerate(reversed(self.preview_history), 1):
                            f.write(f"{i}. {item['filename']} - {item['timestamp'].toString()}\n")
                
                QMessageBox.information(self, "Success", f"Report exported to {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export report: {str(e)}")
    
    def refresh_preview(self):
        """Refresh current preview"""
        if self.current_sample:
            self.update_sample_display()
    
    def generate_batch_preview(self):
        """Generate batch preview grid"""
        # TODO: Implement batch preview generation
        QMessageBox.information(self, "Info", "Batch preview generation not yet implemented.")
    
    def update_history_count(self):
        """Update history count label"""
        count = len(self.preview_history)
        self.history_count_label.setText(f"{count} items in history")
    
    def scroll_history_to_bottom(self):
        """Scroll history to show latest items"""
        scrollbar = self.history_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.minimum())  # Scroll to top (newest items)