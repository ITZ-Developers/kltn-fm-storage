from fastapi import FastAPI, Request, Depends, HTTPException, Header, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uuid, os, base64
from typing import Optional
import mimetypes

from face_utils import extract_embedding, save_embedding, match_embedding, validate_api_key
from rag_chatbot import load_and_split_pdf, create_vector_db, get_qa_chain

PDF_FILE_PATH = "uploaded.pdf"

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


class Question(BaseModel):
    message: str

@app.post("/v1/rag/upload-pdf")
async def upload(file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    contents = await file.read()
    with open("uploaded.pdf", "wb") as f:
        f.write(contents)
    docs = load_and_split_pdf("uploaded.pdf")
    create_vector_db(docs)
    return response(True, "Processed and saved vector successfully")

@app.post("/v1/rag/ask")
def ask_bot(q: Question, api_key: str = Depends(verify_api_key)):
    if not q.message:
        return response(False, "Question is required")
    try:
        chain = get_qa_chain()
        result = chain.run(q.message)
        return response(
            True,
            "Request success",
            data={"answer": result}
        )
    except Exception:
        return response(False, "No data to ask")
    

@app.get("/v1/rag/check-pdf")
async def check_status(api_key: str = Depends(verify_api_key)):
    if os.path.exists(PDF_FILE_PATH):
        return response(True, "PDF file is uploaded")
    else:
        return response(True, "No PDF file uploaded")
    
@app.get("/v1/rag/download-pdf")
async def download_pdf(api_key: str = Depends(verify_api_key)):
    if not os.path.exists(PDF_FILE_PATH):
        raise HTTPException(status_code=404, detail="PDF file not found")
    mime_type, _ = mimetypes.guess_type(PDF_FILE_PATH)
    return FileResponse(PDF_FILE_PATH, media_type=mime_type)