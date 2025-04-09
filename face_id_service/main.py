from fastapi import FastAPI, Request, Depends, HTTPException, Header
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid, os, base64
from typing import Optional

from face_utils import extract_embedding, save_embedding, match_embedding, validate_api_key

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def response(result: bool, message: str, data: dict = None) -> dict:
    return {"result": result, "message": message, "data": data or {}}

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if not validate_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return x_api_key

@app.get("/", response_class=HTMLResponse)
async def webcam_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ImagePayload(BaseModel):
    image_data: str
    user_id: str = None

def save_base64_image(image_data: str) -> str:
    header, encoded = image_data.split(",", 1)
    binary = base64.b64decode(encoded)
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(binary)
    return fpath

@app.post("/register_webcam")
async def register_webcam(data: ImagePayload, api_key: str = Depends(verify_api_key)):
    if not data.user_id or  not data.image_data:
        return response(False, "User ID and image data are required")
    path = save_base64_image(data.image_data)
    embedding = extract_embedding(path)
    os.remove(path)

    if embedding is None:
        return response(False, "No face found")

    save_embedding(data.user_id, embedding)
    return response(True, f"Registered {data.user_id} success")

@app.post("/verify_webcam")
async def verify_webcam(data: ImagePayload, api_key: str = Depends(verify_api_key)):
    path = save_base64_image(data.image_data)
    embedding = extract_embedding(path)
    os.remove(path)

    if embedding is None:
        return response(False, "No face found")

    user_id, confidence = match_embedding(embedding)
    if user_id:
        return response(
            True,
            f"Verify success: {user_id}",
            data={"user_id": user_id, "conf": round(confidence, 2)}
        )
    return response(False, "No match found")
