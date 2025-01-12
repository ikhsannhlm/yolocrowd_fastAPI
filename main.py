from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import torch
from models.yolo import Model  # Impor model custom dari struktur folder
from utils.general import non_max_suppression, scale_coords  # Fungsi utilitas
from PIL import Image
import io
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("uvicorn")


app = FastAPI()

# Load YOLO model
model_path = "models/yolo-crowd.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memuat checkpoint, yang berisi model dan metadata
try:
    checkpoint = torch.load(model_path, map_location=device)
    model = checkpoint['model']  # Mendapatkan model yang sebenarnya
    model.eval()  # Set model ke mode evaluasi
    model_names = checkpoint.get('names', None)
    if model_names is None:
        model_names = ['class_0']  # Sesuaikan ini dengan label kelas yang Anda gunakan
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Response model untuk deteksi
class DetectionResult(BaseModel):
    label: str
    confidence: float
    bbox: list

@app.post("/predict", response_model=list[DetectionResult])
async def predict(file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi objek pada gambar yang diunggah.
    """
    try:
        # Baca file gambar
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize gambar ke ukuran yang diinginkan oleh model (misal: 640x640)
        image = image.resize((640, 640))  # Ukuran ini bisa disesuaikan dengan ukuran input model Anda

        # Konversi gambar ke format tensor
        img = np.array(image)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Ubah tensor menjadi half precision (FP16) jika model menggunakan HalfTensor
        img_tensor = img_tensor.half()  # Menggunakan half precision

        # Inferensi menggunakan model
        pred = model(img_tensor)[0]
        detections = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]  # Post-processing

        # Buat respons
        response = []
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                response.append({
                    "label": model_names[int(cls)],  # Mengambil nama kelas berdasarkan indeks
                    "confidence": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                })
        return JSONResponse(content=response)


    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.get("/")
async def root():
    """
    Endpoint root untuk mengecek status API.
    """
    return {"message": "YOLO Crowd API is running!"}
