from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
from PIL import Image
import io
from .utils import get_transforms, load_checkpoint
import segmentation_models_pytorch as smp

app = FastAPI(title="Food Segmentation API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = smp.DeepLabV3Plus(encoder_name="resnet50", classes=104)
model = load_checkpoint(model, "models/mejor_modelo_B.pth", DEVICE)
transforms = get_transforms()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Leer imagen
    request_object_content = await file.read()
    img = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    img_np = np.array(img)

    # 2. Preprocesar
    augmented = transforms(image=img_np)
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    # 3. Inferir
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.argmax(1).cpu().numpy()[0]

    # 4. Retornar clases detectadas
    detected_classes = np.unique(mask).tolist()
    return {"detected_ingredients_ids": detected_classes}