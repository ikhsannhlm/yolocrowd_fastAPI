from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from models.yolo import Model  # Impor model custom dari struktur folder
from utils.general import non_max_suppression  # Fungsi utilitas
from PIL import Image, ImageDraw
import numpy as np
import io
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


# Fungsi untuk menyimpan gambar dalam bentuk file sementara
def save_image(image: Image.Image) -> io.BytesIO:
    img_io = io.BytesIO()
    image.save(img_io, format='PNG')
    img_io.seek(0)  # Kembali ke awal file untuk dibaca
    return img_io


# Response model untuk deteksi
class DetectionResult(BaseModel):
    total_people_detected: int


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint untuk mendeteksi jumlah orang pada gambar yang diunggah,
    serta mengembalikan gambar yang sudah diberi bounding box.
    """
    try:
        # Baca file gambar
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Resize gambar ke ukuran yang diinginkan oleh model (misal: 640x640)
        image_resized = image.resize((640, 640))  # Ukuran ini bisa disesuaikan dengan ukuran input model Anda

        # Konversi gambar ke format tensor
        img = np.array(image_resized)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Ubah tensor menjadi half precision (FP16) jika model menggunakan HalfTensor
        img_tensor = img_tensor.half()  # Menggunakan half precision

        # Inferensi menggunakan model
        pred = model(img_tensor)[0]
        detections = non_max_suppression(pred, conf_thres=0.45, iou_thres=0.45)[0]  # Post-processing

        # Hitung jumlah orang yang terdeteksi dan gambar bounding box
        person_count = 0  # Variabel untuk menghitung jumlah orang
        draw = ImageDraw.Draw(image_resized)  # Gambar untuk menggambar bounding box

        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                label = model_names[int(cls)]  # Mengambil nama kelas berdasarkan indeks

                # Jika labelnya "person" (misal class_0 adalah orang), hitung jumlah orang
                if label == "class_0":  # Gantilah dengan label yang sesuai untuk orang
                    person_count += 1

                    # Menggambar bounding box pada gambar
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Simpan gambar yang sudah diberi bounding box dalam file sementara
        img_io = save_image(image_resized)

        # Kembalikan jumlah orang yang terdeteksi dalam JSON response
        return StreamingResponse(
            img_io,
            media_type="image/png",  # Atau "image/jpeg" jika Anda ingin menggunakan JPEG
            headers={"X-Total-People-Detected": str(person_count)}  # Menambahkan header jumlah orang
        )

    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")


@app.get("/")
async def root():
    """
    Endpoint root untuk mengecek status API.
    """
    return {"message": "YOLO Crowd API is running!"}
