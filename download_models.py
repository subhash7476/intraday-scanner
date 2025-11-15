# download_models.py
import os
import urllib.request

os.makedirs("models", exist_ok=True)

models = {
    "xgboost_intraday.pkl": "https://grok.x.ai/models/xgboost_intraday.pkl",
    "cnn_model.h5": "https://grok.x.ai/models/cnn_model.h5",
    "lstm_model.pt": "https://grok.x.ai/models/lstm_model.pt"
}

for name, url in models.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"  → {path}")
    else:
        print(f"Already exists: {name}")

print("\nAll models ready in ./models/")
