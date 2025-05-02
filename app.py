from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the model ONCE
model = load_model("models/pneu_cnn_model.h5")

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    imagefile = request.files["image"]
    img = load_img(imagefile, target_size=(500, 500), color_mode='grayscale')
    x = img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    prediction = model.predict(x)
    score = float(prediction[0][0])
    result = 'Positive' if score >= 0.5 else 'Negative'
    confidence = round(score * 100, 2)

    return jsonify({
        'prediction': result,
        'confidence_percent': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
