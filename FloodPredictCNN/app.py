from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
from cnn_model import CNNModel  # <-- Only model, not training code
import os

app = Flask(__name__)

# Load the trained model (weights only)
model = CNNModel()
model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        img = Image.open(file).convert('RGB')
        img = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = 'Flooded' if predicted.item() == 0 else 'Non-Flooded'
        print(f"Prediction: {result}")

        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500




