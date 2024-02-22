import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.blip_vqa import blip_vqa

# Weights are loaded internally
print("Loading models...")
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
image_size = 480

model = blip_vqa(med_config="configs/med_config.json", pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model.to(device)

print("Models loaded.")

# Set up FastAPI server
from .routes import router

fast_api_app = FastAPI()
fast_api_app.add_middleware(
    middleware_class=CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
fast_api_app.include_router(router)
