from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

router = APIRouter(
    prefix="/ai",
    tags=["AI Detection"]
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}


@router.post("/predict")
async def predict_damage(file: UploadFile = File(...)):

    # Validate file
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are allowed"
        )

    # Generate unique filename
    extension = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{extension}"

    file_path = UPLOAD_DIR / filename

    # Save uploaded image
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Temporary AI result
    # This will be replaced by the real YOLO model later.
    prediction = {
        "damage_type": "Pothole",
        "confidence": 0.96,
        "severity": "High"
    }

    return {
        "message": "Image analyzed successfully",
        "image_path": str(file_path),
        "prediction": prediction
    }