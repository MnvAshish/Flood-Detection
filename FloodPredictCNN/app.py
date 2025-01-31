from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os

# Import your trained CNN model (ensure cnntrain.py contains the model definition)
from cnntrain import CNNModel

# Initialize Flask app
app = Flask(__name__)

# Load your trained model correctly
model = CNNModel()  # Create an instance of the model
model.load_state_dict(torch.load('cnn_model.pth', map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set model to evaluation mode

# Define preprocessing for uploaded images (no augmentations)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Corrected normalization
])

@app.route('/')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Preprocess the image
        img = Image.open(file).convert('RGB')  # Ensure RGB format
        img = transform(img).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            result = 'Flooded' if predicted.item() == 0 else 'Non-Flooded'  # Ensure correct label mapping

        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ensure the app runs correctly
if __name__ == "__main__":
    app.run(debug=True)
