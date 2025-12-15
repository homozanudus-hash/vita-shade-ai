from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import json
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F
import io
import os
import requests

MODEL_PATH = "vita_shade_model.pth"
MODEL_URL = "https://drive.google.com/uc?id=189uRgY0P6WgBmYa6mlKg6S0ZtkpyIxe7"

# если модели нет — скачиваем
if not os.path.exists(MODEL_PATH):
    print("Скачиваем модель из Google Drive...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import json
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F
import io

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("classes.json") as f:
    classes = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("vita_shade_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = F.softmax(output, dim=1)[0]

    result = {
        classes[i]: round(probs[i].item()*100, 2)
        for i in range(len(classes))
    }

    best = max(result, key=result.get)

    return {
        "predicted_shade": best,
        "confidence": result[best],
        "all_probabilities": result
    }
