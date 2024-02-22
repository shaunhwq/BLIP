import json

import torch
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter
from PIL import Image

from utils.image_preprocess import load_image
from . import conversions
from . import model, device, image_size

router = APIRouter()


@router.get("/ping", response_class=JSONResponse)
async def ping():
    return {"message": "pong"}


@router.post("/predict", response_class=JSONResponse)
async def predict(request: Request):
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
