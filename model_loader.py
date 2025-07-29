import os
import gdown  # pip install gdown

# === Constants ===
MODEL_ID = "1N5KrdeHA7rcJBN-qnhES17ylUUdjltOm"  # From your shared Drive link
MODEL_DIR = "models"  # This is your local folder to save the model
MODEL_FILENAME = "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        print(f"⬇️ Downloading model to {MODEL_PATH} from Google Drive...")
        gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully.")
    else:
        print("✅ Model already exists locally:", MODEL_PATH)

    return MODEL_PATH  # Return absolute path for llama-cpp

if __name__ == "__main__":
    model_path = download_model()
    print("📦 Model ready at:", model_path)
