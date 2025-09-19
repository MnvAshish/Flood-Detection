from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
from cnn_model import CNNModel  # your model class
import os
from huggingface_hub import hf_hub_download

app = Flask(__name__)

# CONFIG: change these
HF_REPO_ID = "mnvashish/Flood-detection"   # e.g. "ashish/flood-cnn"
HF_FILENAME = "cnn_model.pth"                    # file name you uploaded to HF
HF_TOKEN_ENV_VARS = ("HUGGINGFACE_HUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")

# Helper to get token (None if public model)
HF_TOKEN = None
for env in HF_TOKEN_ENV_VARS:
    if os.getenv(env):
        HF_TOKEN = os.getenv(env)
        break

# where to cache downloaded model
MODEL_CACHE_DIR = "hf_models"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
local_model_path = os.path.join(MODEL_CACHE_DIR, HF_FILENAME)

def download_model_from_hf():
    global local_model_path
    if os.path.exists(local_model_path):
        print("Using cached model:", local_model_path)
        return local_model_path
    try:
        print(f"Downloading {HF_FILENAME} from {HF_REPO_ID} ...")
        # hf_hub_download returns the path to the downloaded file
        local_model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model",
            token=HF_TOKEN,
            cache_dir=MODEL_CACHE_DIR
        )
        print("Downloaded model to:", local_model_path)
        return local_model_path
    except Exception as e:
        print("Failed to download model from Hugging Face Hub:", str(e))
        raise

# Load model (download if needed)
model = CNNModel()
try:
    path = download_model_from_hf()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    # Fallback: you can still try local path if you have one
    print("Error loading model:", e)
    raise

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
