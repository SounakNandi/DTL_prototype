from flask import Flask, request, render_template, jsonify
import os
import random
#from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#def load_model():
    #return keras.models.load_model("dog_model.h5")
#model = load_model()

# Simulated disease list
DISEASES = ["Demodicosis", "Hypersensitivity", "Fungal Infections", "Dermatitis", "Ringworm", "Healthy"]

@app.route('/')
def index():
    return render_template("index.html")  # Render the form for image upload

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the image
    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    # Simulate prediction (use your actual model here instead of random)
    predicted_disease = random.choice(DISEASES)
    confidence = round(random.uniform(70, 99), 2)

    # Return prediction as JSON response
    return jsonify({
        'disease': predicted_disease,
        'confidence': confidence,
        'filename': image.filename
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
