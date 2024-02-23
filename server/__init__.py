import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from threading import Timer

import torch
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from models.blip_vqa import blip_vqa
from utils.image_preprocess import load_image
from . import conversions


# Weights are loaded internally
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
image_size = 480

model = None


# Start server first
def load_model():
    global model, model_loaded
    print("Loading models...")
    model = blip_vqa(med_config="configs/med_config.json", pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model.to(device)
    print("Models loaded.")


t = Timer(2.0, load_model)
t.start()

fast_api_app = FastAPI()
fast_api_app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@fast_api_app.get("/ping", response_class=JSONResponse)
async def ping():
    return {"message": "pong"}


@fast_api_app.post("/predict", response_class=JSONResponse)
async def predict(request: Request):
    if model is None:
        return JSONResponse({"error": "model not loaded"}, status_code=500)

    # Retrieve image and question from body
    body = await request.body()
    json_obj = json.loads(body.decode("utf-8"))
    image_str = json_obj.get("image", None)
    question = json_obj.get("question", None)

    if image_str is None or question is None:
        return JSONResponse({"Make sure input has fields 'image' and 'question'"}, status_code=500)
    image = conversions.b64str_to_cv2_image(image_str)
    image = Image.fromarray(image).convert('RGB')
    in_tensor = load_image(image, image_size, device)

    with torch.no_grad():
        answer = model(in_tensor, question, train=False, inference='generate')

    return {"answer": answer[0]}

@fast_api_app.get('/model-status')
async def get_model_status():
    return {"model_loaded": model is not None}
