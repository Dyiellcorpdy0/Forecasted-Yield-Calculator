import sys
import torch
import torch.nn as nn
import joblib
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QDoubleSpinBox, QFormLayout, QFrame, QMessageBox, 
                             QSpinBox, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

class YieldNet(nn.Module):
    def __init__(self, input_size):
        super(YieldNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

class YieldCalculatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Crop Predictor PRO")
        self.setGeometry(100, 100, 1000, 550)
        
        # Обновленный QSS: исправлен черный текст в QComboBox, убраны рамки, улучшены шрифты
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f111a;
            }
            QLabel {
                color: #e2e8f0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 15px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #1e212b;
                color: #ffffff;
                border: 1px solid #2d313f;
                border-radius: 6px;
                padding: 10px;
                font-size: 15px;
            }
            QComboBox::drop-down {
                border: none;
            }
            /* Исправление выпадающего списка (чтобы текст был белым, а не черным) */
            QComboBox QAbstractItemView {
                background-color: #1e212b;
                color: #ffffff;
                selection-background-color: #3b82f6;
                selection-color: #ffffff;
                border: 1px solid #2d313f;
                outline: none;
            }
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #3b82f6;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                padding: 16px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)

        self.load_ai_models()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(25, 25, 25, 25)
        self.main_layout.setSpacing(25)

        # Функция для создания панелей с легкой тенью
        def create_panel():
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #161925;
                    border: 1px solid #232736;
                    border-radius: 12px;
                }
            """)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QColor(0, 0, 0, 100))
            shadow.setOffset(0, 5)
            frame.setGraphicsEffect(shadow)
            return frame

        # --- ЛЕВАЯ ПАНЕЛЬ (Ввод параметров) ---
        self.left_frame = create_panel()
        self.left_layout = QVBoxLayout(self.left_frame)
        self.left_layout.setContentsMargins(30, 30, 30, 30)
        self.left_layout.setSpacing(20)

        title_lbl = QLabel("Ввод параметров")
        title_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #ffffff; border: none;")
        self.left_layout.addWidget(title_lbl)

        self.form_layout = QFormLayout()
        self.form_layout.setVerticalSpacing(15)

        self.combo_country = QComboBox()
        self.combo_country.addItems(['Belarus', 'Ukraine', 'Poland', 'Lithuania', 'Latvia', 'Germany', 'Canada', 'Russian Federation'])
        
        self.combo_crop = QComboBox()
        self.combo_crop.addItems(['Potatoes', 'Wheat'])

        self.spin_year = QSpinBox(); self.spin_year.setRange(2024, 2050); self.spin_year.setValue(2024)
        
        # Обновленные стартовые значения
        self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(-20, 50); self.spin_temp.setValue(10.0); self.spin_temp.setSuffix(" °C")
        self.spin_rain = QDoubleSpinBox(); self.spin_rain.setRange(0, 3000); self.spin_rain.setValue(650.0); self.spin_rain.setSuffix(" мм")
        self.spin_pest = QDoubleSpinBox(); self.spin_pest.setRange(0, 100000); self.spin_pest.setValue(1000.0); self.spin_pest.setSuffix(" тонн")

        self.form_layout.addRow("Страна:", self.combo_country)
        self.form_layout.addRow("Культура:", self.combo_crop)
        self.form_layout.addRow("Год:", self.spin_year)
        self.form_layout.addRow("Температура:", self.spin_temp)
        self.form_layout.addRow("Осадки:", self.spin_rain)
        self.form_layout.addRow("Пестициды:", self.spin_pest)

        for i in range(self.form_layout.rowCount()):
            self.form_layout.itemAt(i, QFormLayout.LabelRole).widget().setStyleSheet("border: none; color: #94a3b8;")

        self.left_layout.addLayout(self.form_layout)
        self.left_layout.addStretch()

        self.btn_calc = QPushButton("СГЕНЕРИРОВАТЬ ПРОГНОЗ")
        self.btn_calc.clicked.connect(self.calculate_ai_yield)
        self.left_layout.addWidget(self.btn_calc)
        
        self.main_layout.addWidget(self.left_frame, stretch=1)

        # --- ПРАВАЯ ПАНЕЛЬ (Результат со спецэффектами) ---
        self.right_frame = create_panel()
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setContentsMargins(20, 40, 20, 40)

        res_title = QLabel("Прогнозируемая урожайность")
        res_title.setStyleSheet("font-size: 18px; color: #94a3b8; border: none; letter-spacing: 1px;")
        res_title.setAlignment(Qt.AlignCenter)
        
        # Главная цифра
        self.label_result_value = QLabel("---")
        self.label_result_value.setAlignment(Qt.AlignCenter)
        self.label_result_value.setFont(QFont("Segoe UI", 75, QFont.Bold)) # Сделал шрифт еще больше
        self.label_result_value.setStyleSheet("color: #ffffff; border: none; margin-top: 20px; margin-bottom: 10px;") 
        
        # Добавляем спецэффект: Неоновое свечение (Glow) для текста
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(35) # Сила размытия свечения
        self.glow_effect.setColor(QColor(59, 130, 246, 150)) # Синий неон по умолчанию
        self.glow_effect.setOffset(0, 0) # Свечение ровно со всех сторон
        self.label_result_value.setGraphicsEffect(self.glow_effect)
        
        self.label_unit = QLabel("ТОНН НА ГЕКТАР")
        self.label_unit.setAlignment(Qt.AlignCenter)
        self.label_unit.setStyleSheet("color: #64748b; font-size: 14px; font-weight: bold; border: none; letter-spacing: 2px;")

        self.right_layout.addStretch()
        self.right_layout.addWidget(res_title)
        self.right_layout.addWidget(self.label_result_value)
        self.right_layout.addWidget(self.label_unit)
        self.right_layout.addStretch()

        self.main_layout.addWidget(self.right_frame, stretch=1)

    def load_ai_models(self):
        try:
            self.scaler = joblib.load("scaler.pkl")
            self.le_area = joblib.load("le_area.pkl")
            self.le_item = joblib.load("le_item.pkl")
            
            self.model = YieldNet(input_size=6)
            self.model.load_state_dict(torch.load("yield_model.pth"))
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не найдены файлы модели!\n{e}")
            sys.exit()

    def calculate_ai_yield(self):
        try:
            country = self.combo_country.currentText()
            crop = self.combo_crop.currentText()
            year = self.spin_year.value()
            rain = self.spin_rain.value()
            pest = self.spin_pest.value()
            temp = self.spin_temp.value()

            area_encoded = self.le_area.transform([country])[0]
            item_encoded = self.le_item.transform([crop])[0]

            # Порядок как при обучении!
            X_input = np.array([[year, rain, pest, temp, area_encoded, item_encoded]])
            X_scaled = self.scaler.transform(X_input)
            
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                prediction = self.model(X_tensor)
            
            predicted_yield = max(0, prediction.item())

            # Обновляем текст
            self.label_result_value.setText(f"{predicted_yield:.2f}")
            
            # Динамическое изменение цвета и свечения (Спецэффект реагирует на результат)
            if predicted_yield > 25: 
                # Синий (Отличный урожай - картошка)
                self.label_result_value.setStyleSheet("color: #60a5fa; border: none; margin-top: 20px; margin-bottom: 10px;") 
                self.glow_effect.setColor(QColor(96, 165, 250, 180)) 
            elif predicted_yield < 10: 
                # Красный (Плохой урожай)
                self.label_result_value.setStyleSheet("color: #f87171; border: none; margin-top: 20px; margin-bottom: 10px;")
                self.glow_effect.setColor(QColor(248, 113, 113, 180))
            else:
                # Зеленый (Средний нормальный урожай)
                self.label_result_value.setStyleSheet("color: #4ade80; border: none; margin-top: 20px; margin-bottom: 10px;")
                self.glow_effect.setColor(QColor(74, 222, 128, 180))

        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчета", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    window = YieldCalculatorApp()
    window.show()
    sys.exit(app.exec_())
