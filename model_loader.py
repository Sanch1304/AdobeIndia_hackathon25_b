import os
import gdown  # install with: pip install gdown
import torch  # or tensorflow, etc.

MODEL_ID = ''
MODEL_FILENAME = 'm'  # or .pt, .h5, etc.

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        print("Model not found locally. Downloading from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_FILENAME, quiet=False)
    else:
        print("Model already exists locally.")

def load_model():
    download_model()
    model = torch.load(MODEL_FILENAME, map_location=torch.device('cpu'))  # or custom loader
    model.eval()
    return model
