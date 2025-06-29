import sys
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                             QHBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox, 
                             QProgressBar, QCheckBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt
from config import MODELS_CONFIG, REQUIRED_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN
from utils.metrics import compute_metrics
from utils.plots import create_model_visualizations, plot_comparative_metrics
import importlib

def get_model_instance(model_name):
    """Dynamically imports and returns an instance of the specified model."""
    config = MODELS_CONFIG[model_name]
    module_path, class_name = config["class"].rsplit('.', 1)
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        if config["type"] == "sklearn":
            return model_class(**config.get("params", {}))
        else: # Custom models
            return model_class()
            
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Could not load model: {model_name}. Error: {e}")

class AquaAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AquaAI - Çoklu Model Su Kalitesi Tahmin Sistemi")
        self.setMinimumSize(1000, 700)
        
        self.training_data = None
        self.test_data = None
        self.all_metrics = {}
        self.model_checkboxes = {}
        
        self.init_ui()
    
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        title = QLabel("AquaAI - Çoklu Model Su Kalitesi Tahmin Sistemi")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        data_layout = QHBoxLayout()
        self.train_btn = QPushButton("Eğitim Verisi Yükle")
        self.test_btn = QPushButton("Test Verisi Yükle")
        
        for btn in [self.train_btn, self.test_btn]:
            btn.setMinimumHeight(50)
            btn.setStyleSheet("""
                QPushButton { background-color: #2196F3; color: white; border-radius: 5px; font-size: 16px; padding: 10px; }
                QPushButton:hover { background-color: #1976D2; }
            """)
            data_layout.addWidget(btn)
        layout.addLayout(data_layout)
        
        model_group = QGroupBox("Analiz Edilecek Modelleri Seçin")
        model_group.setStyleSheet("""
            QGroupBox { font-size: 16px; font-weight: bold; border: 2px solid #ccc; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        """)
        
        model_layout = QGridLayout()
        
        # Dynamically create checkboxes from config
        for i, (model_key, config) in enumerate(MODELS_CONFIG.items()):
            checkbox = QCheckBox(config["name"])
            checkbox.setStyleSheet("""
                QCheckBox { font-size: 14px; padding: 5px; }
                QCheckBox::indicator { width: 20px; height: 20px; }
            """)
            model_layout.addWidget(checkbox, i // 4, i % 4)
            self.model_checkboxes[model_key] = checkbox
            
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        select_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Tümünü Seç")
        self.deselect_all_btn = QPushButton("Tümünü Kaldır")
        
        for btn in [self.select_all_btn, self.deselect_all_btn]:
            btn.setStyleSheet("""
                QPushButton { background-color: #FF9800; color: white; border-radius: 5px; font-size: 14px; padding: 8px; }
                QPushButton:hover { background-color: #F57C00; }
            """)
            select_layout.addWidget(btn)
        layout.addLayout(select_layout)
        
        self.run_analysis_btn = QPushButton("Seçili Modelleri Analiz Et")
        self.run_analysis_btn.setMinimumHeight(60)
        self.run_analysis_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; border-radius: 8px; font-size: 18px; font-weight: bold; padding: 15px; }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        layout.addWidget(self.run_analysis_btn)
        
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        self.status_label = QLabel("Hazır")
        self.status_label.setStyleSheet("font-size: 14px; color: #666; margin: 10px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.train_btn.clicked.connect(self.load_training_data)
        self.test_btn.clicked.connect(self.load_test_data)
        self.select_all_btn.clicked.connect(lambda: self.toggle_all_models(True))
        self.deselect_all_btn.clicked.connect(lambda: self.toggle_all_models(False))
        self.run_analysis_btn.clicked.connect(self.run_selected_analyses)

    def toggle_all_models(self, checked):
        for checkbox in self.model_checkboxes.values():
            checkbox.setChecked(checked)

    def load_data(self, data_type):
        file_path, _ = QFileDialog.getOpenFileName(self, f"{data_type.capitalize()} Verisi Seç", "", 
                                                 "Excel Files (*.xlsx *.xls);;CSV Files (*.csv)")
        if file_path:
            try:
                data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                
                required = REQUIRED_COLUMNS[data_type]
                missing = [col for col in required if col not in data.columns]
                
                if missing:
                    raise ValueError(f"Eksik sütunlar: {', '.join(missing)}")
                
                if data_type == 'train':
                    self.training_data = data
                else:
                    self.test_data = data
                    
                self.status_label.setText(f"{data_type.capitalize()} verisi yüklendi: {len(data)} satır")
                QMessageBox.information(self, "Başarılı", f"{data_type.capitalize()} verisi başarıyla yüklendi!")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Veri yüklenirken hata oluştu: {str(e)}")

    def load_training_data(self):
        self.load_data('train')

    def load_test_data(self):
        self.load_data('test')

    def run_selected_analyses(self):
        if self.training_data is None or self.test_data is None:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce eğitim ve test verilerini yükleyin!")
            return
        
        selected_models = [key for key, cb in self.model_checkboxes.items() if cb.isChecked()]
        
        if not selected_models:
            QMessageBox.warning(self, "Uyarı", "Lütfen en az bir model seçin!")
            return
        
        self.progress.setVisible(True)
        self.progress.setMaximum(len(selected_models))
        self.progress.setValue(0)
        
        X_train = self.training_data[FEATURE_COLUMNS].values
        y_train = self.training_data[TARGET_COLUMN].values
        X_test = self.test_data[FEATURE_COLUMNS].values
        y_test = self.test_data[TARGET_COLUMN].values
        
        completed_models, failed_models = [], []
        self.all_metrics = {}
        
        for i, model_key in enumerate(selected_models):
            try:
                self.status_label.setText(f"{MODELS_CONFIG[model_key]['name']} analiz ediliyor...")
                self.progress.setValue(i)
                QApplication.processEvents()
                
                self.analyze_model(model_key, X_train, y_train, X_test, y_test)
                completed_models.append(MODELS_CONFIG[model_key]['name'])
                
            except Exception as e:
                failed_models.append(f"{MODELS_CONFIG[model_key]['name']}: {str(e)}")
                print(f"Hata - {model_key}: {str(e)}")
        
        self.progress.setVisible(False)
        self.progress.setValue(len(selected_models))
        
        if len(completed_models) > 1:
            try:
                plot_comparative_metrics(self.all_metrics, "results/comparative_metrics.pdf")
            except Exception as e:
                print(f"Comparative metrics plot error: {e}")
        
        if completed_models:
            QMessageBox.information(self, "Başarılı", 
                                  f"Analiz tamamlandı!\n\nBaşarılı: {', '.join(completed_models)}\n\n"
                                  f"Sonuçlar 'results/' klasörüne kaydedildi.")
        
        if failed_models:
            QMessageBox.warning(self, "Uyarı", "Başarısız olan modeller:\n" + "\n".join(failed_models))

    def analyze_model(self, model_key, X_train, y_train, X_test, y_test):
        output_dir = f"results/{model_key.lower()}"
        os.makedirs(output_dir, exist_ok=True)
        
        model_config = MODELS_CONFIG[model_key]
        model = get_model_instance(model_key)
        
        if model_config["type"] == "custom":
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = model.evaluate(y_test, y_pred)
        else: # sklearn
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            
        self.all_metrics[model_config["name"]] = metrics
        
        with open(f"{output_dir}/metrics.txt", "w", encoding='utf-8') as f:
            f.write(f"{model_config['name']} Model Performans Metrikleri\n")
            f.write("=" * 40 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        create_model_visualizations(model_key, model, X_train, y_train, X_test, y_test, y_pred, output_dir)

def main():
    app = QApplication(sys.argv)
    window = AquaAI()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()