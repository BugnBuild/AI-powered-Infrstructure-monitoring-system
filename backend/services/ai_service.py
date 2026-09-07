from pathlib import Path
from ultralytics import YOLO


# Path to the trained YOLO model
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "runs" / "detect" / "road_damage_ai" / "weights" / "best.pt"

# Load model once when the backend starts
model = YOLO(str(MODEL_PATH))


CLASS_NAMES = {
    0: "Crack",
    1: "Pothole",
    2: "Surface Erosion"
}


def analyze_image(image_path: str):
    """
    Analyze a road-damage image using the trained YOLO model.
    """

    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # No objects detected
    if result.boxes is None or len(result.boxes) == 0:
        return {
            "damage_type": "No damage detected",
            "confidence": 0.0,
            "severity": "Low"
        }

    # Find detection with highest confidence
    best_index = result.boxes.conf.argmax().item()

    class_id = int(result.boxes.cls[best_index].item())
    confidence = float(result.boxes.conf[best_index].item())

    damage_type = CLASS_NAMES.get(
        class_id,
        "Unknown damage"
    )

    # Simple severity logic based on confidence
    if confidence >= 0.75:
        severity = "High"
    elif confidence >= 0.50:
        severity = "Medium"
    else:
        severity = "Low"

    return {
        "damage_type": damage_type,
        "confidence": confidence,
        "severity": severity
    }