from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from pathlib import Path
import shutil
import uuid
import random

from database import get_db
from models.complaint import Complaint
from services.ai_service import analyze_image

router = APIRouter(
    prefix="/complaints",
    tags=["Complaints"]
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Dataset auto-classification folder ───────────────────────────────────────
# Lives at: <project_root>/datasets/
DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

# Damage type → safe folder name mapping
DAMAGE_FOLDER = {
    "Crack":           "Crack",
    "Pothole":         "Pothole",
    "Surface Erosion": "Surface_Erosion",
}

# 70 / 20 / 10 split — random assignment per submission
SPLIT_WEIGHTS = [
    ("train", 0.70),
    ("val",   0.20),
    ("test",  0.10),
]


def pick_split() -> str:
    """Randomly assign a split according to 70/20/10 weights."""
    r = random.random()
    cumulative = 0.0
    for name, weight in SPLIT_WEIGHTS:
        cumulative += weight
        if r < cumulative:
            return name
    return "train"


def save_to_dataset(
    source_path: Path,
    damage_type: str,
    complaint_id: int
) -> str:
    """
    Copy the uploaded image into:
      datasets/<split>/images/<DamageClass>/<complaint_id>_<filename>

    Returns the relative dataset path (for logging/debugging).
    """
    folder_name = DAMAGE_FOLDER.get(damage_type, "Crack")
    split = pick_split()

    dest_dir = DATASETS_DIR / split / "images" / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_filename = f"{complaint_id}_{source_path.name}"
    dest_path = dest_dir / dest_filename

    shutil.copy2(source_path, dest_path)
    return str(dest_path.relative_to(DATASETS_DIR.parent))


# ─────────────────────────────────────────────────────────────────────────────

@router.post("/submit")
async def submit_complaint(
    user_id: int = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    locality: str = Form(default="Unknown"),   # ← locality from frontend GPS
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Validate image type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are allowed"
        )

    # 2. Save to uploads/
    extension = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4()}{extension}"
    file_path = UPLOAD_DIR / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 3. AI analysis
    analysis = analyze_image(str(file_path))
    damage_type = analysis["damage_type"]
    confidence  = analysis["confidence"]
    severity    = analysis["severity"]

    # 4. Build description (includes locality)
    location_label = locality if locality and locality != "Unknown" else (
        f"{latitude:.5f}, {longitude:.5f}"
    )
    description = (
        f"{damage_type} detected at {location_label} "
        f"with {confidence * 100:.0f}% confidence. "
        f"Severity: {severity}. "
        "Immediate inspection is recommended."
    )

    # 5. Create & save complaint record
    complaint = Complaint(
        user_id=user_id,
        image_path=str(file_path),
        damage_type=damage_type,
        severity=severity,
        latitude=latitude,
        longitude=longitude,
        locality=locality,
        description=description,
        status="Submitted"
    )
    db.add(complaint)
    db.commit()
    db.refresh(complaint)

    # 6. Auto-classify into datasets/ folder (70/20/10)
    dataset_path = "skipped"
    if damage_type != "No damage detected":
        dataset_path = save_to_dataset(
            file_path,
            damage_type,
            complaint.id
        )

    return {
        "message": "Complaint submitted successfully",
        "complaint_id": complaint.id,
        "analysis": analysis,
        "location": {
            "latitude": latitude,
            "longitude": longitude,
            "locality": locality
        },
        "dataset_path": dataset_path,
        "image": str(file_path),
        "status": complaint.status
    }


@router.get("/")
def get_complaints(db: Session = Depends(get_db)):
    return db.query(Complaint).all()


@router.get("/{complaint_id}")
def get_complaint(
    complaint_id: int,
    db: Session = Depends(get_db)
):
    complaint = db.query(Complaint).filter(
        Complaint.id == complaint_id
    ).first()

    if not complaint:
        raise HTTPException(
            status_code=404,
            detail="Complaint not found"
        )

    return complaint
