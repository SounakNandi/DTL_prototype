import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QComboBox, QTextEdit, QFrame
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
import tensorflow as tf
import numpy as np

class SkinDiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cat & Dog Skin Disease Predictor")
        self.setFixedSize(1000, 800)  # Fixed clean window size

        self.model_cat = None
        self.model_dog = None
        self.image_path = None

        self.cat_labels = ['Flea_Allergy', 'Healthy', 'Ringworm', 'Scabies']
        self.dog_labels = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'Demodicosis', 'Ringworm']

        self.load_models()
        self.init_ui()

    def load_models(self):
        try:
            self.model_cat = tf.keras.models.load_model("./cat/cat_model.h5")
            print("Cat model loaded successfully.")
        except Exception as e:
            print("Failed to load Cat model:", e)

        try:
            self.model_dog = tf.keras.models.load_model("./dog/dog_model.h5")
            print("Dog model loaded successfully.")
        except Exception as e:
            print("Failed to load Dog model:", e)

    def init_ui(self):
        # Title Label
        self.title_label = QLabel("SKIN DISEASE DETECTOR")
        self.title_label.setFont(QFont('Arial', 30, 1000))
        self.title_label.setAlignment(Qt.AlignCenter)

        # Dropdown
        self.dropdown = QComboBox()
        self.dropdown.addItems(["Cat", "Dog"])
        self.dropdown.setFont(QFont('Arial', 13))
        self.dropdown.setFixedWidth(300)

        # Upload Button
        self.upload_btn = QPushButton("Upload Image")
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setFont(QFont('Arial', 13))
        self.upload_btn.setFixedWidth(300)

        # Buttons layout
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.dropdown)
        btn_layout.addSpacing(50)
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addStretch()

        # Image preview
        self.image_label = QLabel("No image uploaded.")
        self.image_label.setFont(QFont('Arial', 14))
        self.image_label.setAlignment(Qt.AlignCenter)
        # self.image_label.setFixedSize(960, 400)
        self.image_label.setFixedHeight(415)
        self.image_label.setStyleSheet("border: 2px dashed gray;")
        
        # Predict Button
        self.predict_btn = QPushButton("PREDICT")
        self.predict_btn.clicked.connect(self.predict)
        self.predict_btn.setFont(QFont('Arial', 20, 1000))
        self.predict_btn.setFixedWidth(500)

        # Prediction Result Box
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.result_box.setFont(QFont('Arial', 20))
        # self.result_box.setFixedHeight(100)

        # Overall Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        layout.addWidget(self.title_label)
        layout.addLayout(btn_layout)
        layout.addWidget(self.image_label)
        layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_box)

        self.setLayout(layout)

    def upload_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.result_box.clear()

    def predict(self):
        if not self.image_path:
            self.result_box.setText("⚠️ Please upload an image first.")
            return

        selected_model_name = self.dropdown.currentText()

        if selected_model_name == "Cat" and self.model_cat:
            model = self.model_cat
            labels = self.cat_labels
        elif selected_model_name == "Dog" and self.model_dog:
            model = self.model_dog
            labels = self.dog_labels
        else:
            self.result_box.setText(f"⚠️ {selected_model_name} model is not available.")
            return

        img = tf.keras.preprocessing.image.load_img(self.image_path, target_size=(200, 200))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        # img_array /= 255.0

        prediction = model.predict(img_array)

        if prediction.shape[-1] > 1:
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)
        else:
            predicted_class = int(prediction[0][0] > 0.5)
            confidence = prediction[0][0]

        predicted_label = labels[predicted_class]

        self.result_box.setText(
            f"Predicted Disease: {predicted_label}\nConfidence: {confidence * 100:.2f}%"
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SkinDiseasePredictor()
    window.show()
    sys.exit(app.exec_())
