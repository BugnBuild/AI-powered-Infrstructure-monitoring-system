# Smart Civic AI — Data Flow Structure
### AI-Powered Civic Infrastructure Monitoring System
**VIT Bhopal University | Raghwendra Singh · Krish Sachan · Abhishek Singh**

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SMART CIVIC AI PLATFORM                         │
├──────────────────┬──────────────────────┬───────────────────────────────┤
│   FRONTEND       │      BACKEND         │        AI / DATA LAYER        │
│  HTML/CSS/JS     │  FastAPI + Uvicorn   │  YOLOv8 + SQLite + datasets/  │
│  Port: 5500      │  Port: 8000          │  best.pt (trained weights)    │
└──────────────────┴──────────────────────┴───────────────────────────────┘
```

---

## 2. Complete Data Flow — Step by Step

```
CITIZEN ACTION                   SYSTEM PIPELINE                      OUTPUT
─────────────                    ───────────────                      ──────

① Open Website                   Load index.html                      Dashboard rendered
   localhost:5500                 Load style.css + script.js           Complaints fetched
                                  GET /complaints/ → JSON              Table populated
                                         │
                                         ▼
② Click "Report a Problem"        openReport() called                  Modal opens
                                  GPS permission request shown
                                  navigator.geolocation.getCurrentPosition()
                                         │
                                         ▼
③ GPS Permission Granted          Browser returns lat/lon              Spinner shown
                                  reverseGeocode(lat, lon) called      → Nominatim API
                                  OpenStreetMap Nominatim API request  Locality resolved
                                  → "Arera Colony, Bhopal, MP"        Location displayed
                                         │
                                         ▼
④ Upload Road Image               previewImage(event) called           Image previewed
                                  File type validated (JPG/PNG/WEBP)   in modal
                                  Image shown in upload box
                                         │
                                         ▼
⑤ Click "Submit Complaint"        showSubmitConfirmation()             Confirm dialog shown
                                  Locality shown in confirmation        User reviews data
                                         │
                                         ▼
⑥ Click "Confirm Submit"          FormData built:                      Button shows
                                  ├── file    (image)                  "AI Analyzing..."
                                  ├── user_id (1)
                                  ├── latitude
                                  ├── longitude
                                  └── locality
                                  POST /complaints/submit (multipart)
                                         │
                                         ▼
```

---

## 3. Backend Processing Pipeline

```
POST /complaints/submit
        │
        ├─① VALIDATION
        │    content_type ∈ {image/jpeg, image/png, image/webp} ?
        │    → 400 Bad Request if invalid
        │
        ├─② FILE STORAGE
        │    uuid4() filename generated
        │    Image saved → uploads/<uuid>.jpg
        │
        ├─③ AI INFERENCE  ─────────────────────────────────────────────┐
        │    analyze_image(file_path)                                   │
        │         │                                                     │
        │         ▼                                                     │
        │    YOLO model loaded (best.pt)                                │
        │    YOLOv8m.predict(image, conf=0.25)                         │
        │         │                                                     │
        │         ├── Detections found?                                 │
        │         │       YES → extract dominant class                  │
        │         │              class 0 = Crack                        │
        │         │              class 1 = Pothole                      │
        │         │              class 2 = Surface Erosion              │
        │         │              confidence score → float               │
        │         │              bbox → [x1, y1, x2, y2]               │
        │         │                                                     │
        │         └── NO detections → "No damage detected"             │
        │                                                               │
        │    Severity mapped from confidence:                           │
        │         ≥ 0.80 → HIGH                                         │
        │         ≥ 0.55 → MEDIUM                                       │
        │         < 0.55 → LOW                                          │
        │                                                               │
        │    Returns: { damage_type, confidence,                        │
        │               severity, bbox }          ◄─────────────────────┘
        │
        ├─④ DATABASE WRITE (SQLAlchemy → SQLite)
        │    INSERT INTO complaints:
        │    ├── user_id    = 1
        │    ├── image_path = "uploads/<uuid>.jpg"
        │    ├── damage_type= "Pothole"
        │    ├── severity   = "HIGH"
        │    ├── latitude   = 23.2599
        │    ├── longitude  = 77.4126
        │    ├── locality   = "Arera Colony, Bhopal"
        │    ├── description= "Pothole detected at ... 94% confidence"
        │    └── status     = "Submitted"
        │    → complaint.id returned
        │
        └─⑤ DATASET SAVE (70/20/10 split)
             save_to_dataset(file_path, damage_type, complaint_id)
             MD5(filename) % 100:
               0-69  → datasets/train/images/Pothole/<id>_<file>.jpg
               70-89 → datasets/val/images/Pothole/<id>_<file>.jpg
               90-99 → datasets/test/images/Pothole/<id>_<file>.jpg

                         │
                         ▼
              JSON Response returned:
              {
                "message": "Complaint submitted successfully",
                "complaint_id": 42,
                "analysis": {
                  "damage_type": "Pothole",
                  "confidence": 0.942,
                  "severity": "HIGH",
                  "bbox": [x1, y1, x2, y2]
                },
                "location": {
                  "latitude": 23.2599,
                  "longitude": 77.4126,
                  "locality": "Arera Colony, Bhopal"
                },
                "dataset_path": "datasets/train/images/Pothole/42_abc.jpg",
                "status": "Submitted"
              }
```

---

## 4. Frontend Response Handling

```
Response received
       │
       ├── response.ok = true?
       │       YES
       │        ├── closeReport() — modal hidden
       │        ├── resetReportForm() — form cleared
       │        ├── Alert shown:
       │        │     ✅ Complaint Submitted!
       │        │     Damage Type  : Pothole
       │        │     Confidence   : 94.2%
       │        │     Severity     : HIGH
       │        │     Location     : Arera Colony, Bhopal
       │        │     Complaint #  : #42
       │        │     📁 datasets/train/images/Pothole/...
       │        └── loadComplaints() — table refreshed
       │
       └── response.ok = false?
               NO
                └── Alert shown with error detail
                    Form stays open for retry
```

---

## 5. Database Schema

```sql
TABLE: users
┌────────────┬───────────┬───────────────────────────┐
│ Column     │ Type      │ Description               │
├────────────┼───────────┼───────────────────────────┤
│ id         │ INTEGER   │ Primary Key (auto)        │
│ name       │ STRING    │ Full name                 │
│ email      │ STRING    │ Unique, indexed           │
│ password   │ STRING    │ Hashed                    │
│ role       │ STRING    │ "citizen" | "admin"       │
└────────────┴───────────┴───────────────────────────┘

TABLE: complaints
┌─────────────┬───────────┬───────────────────────────┐
│ Column      │ Type      │ Description               │
├─────────────┼───────────┼───────────────────────────┤
│ id          │ INTEGER   │ Primary Key               │
│ user_id     │ INTEGER   │ FK → users.id             │
│ image_path  │ STRING    │ uploads/<uuid>.jpg        │
│ damage_type │ STRING    │ Crack / Pothole / Erosion │
│ severity    │ STRING    │ HIGH / MEDIUM / LOW       │
│ latitude    │ FLOAT     │ GPS lat                   │
│ longitude   │ FLOAT     │ GPS lon                   │
│ locality    │ STRING    │ Reverse-geocoded name     │
│ description │ TEXT      │ Auto-generated AI summary │
│ status      │ STRING    │ Submitted/Reviewing/Done  │
└─────────────┴───────────┴───────────────────────────┘
```

---

## 6. AI Model Data Flow

```
Training Pipeline                       Inference Pipeline
─────────────────                       ──────────────────

Roboflow Dataset (1,071 imgs)           Citizen uploads image
    +                                          │
Kaggle Datasets (19,000+ imgs)          ┌─────▼──────────────┐
    +                                   │  OpenCV imread()    │
img/ folder (18,046 imgs)               │  Resize to 640×640  │
    │                                   │  Normalize pixels   │
    ▼                                   └─────────────────────┘
classify_all_images.py                         │
    │                                          ▼
    ├── YOLO inference per image         YOLOv8m forward pass
    ├── Dominant class assigned          25.9M parameters
    └── 70/20/10 hash split              Single-stage detection
                                               │
datasets/                               ┌──────▼──────────────┐
├── train/ (70%)  ~14,000 imgs          │  Bounding boxes     │
├── val/   (20%)   ~4,000 imgs          │  Class IDs          │
└── test/  (10%)   ~2,000 imgs          │  Confidence scores  │
                                        └─────────────────────┘
    │                                          │
    ▼                                          ▼
4_train_yolo.py                         severity_matrix()
    YOLOv8m.train(                       conf ≥ 0.80 → HIGH
        epochs=100,                      conf ≥ 0.55 → MEDIUM
        patience=20,                     conf < 0.55 → LOW
        imgsz=640,
        optimizer='AdamW'
    )
    │
    ▼
best.pt (trained weights)
    │
    ▼
runs/detect/road_damage_ai/weights/best.pt
models/best.pt  ← FastAPI loads this
```

---

## 7. API Endpoints Reference

```
METHOD   ENDPOINT              DESCRIPTION                  AUTH
──────   ─────────────────     ───────────────────────      ────
GET      /                     Health check                 None
POST     /complaints/submit    Submit + AI analyze          None
GET      /complaints/          List all complaints          None
GET      /complaints/{id}      Single complaint             None
GET      /uploads/{filename}   Serve uploaded image         None
POST     /auth/register        Register user                None
POST     /auth/login           Login + JWT token            None
```

---

## 8. Technology Stack

```
LAYER          TECHNOLOGY              ROLE
─────────────  ──────────────────────  ───────────────────────────────
Frontend       HTML5 + CSS3 + JS ES6   UI, GPS, file upload, dashboard
               Inter (Google Fonts)    Typography
               OpenStreetMap Nominatim Reverse geocoding (free)

Backend        Python 3.13             Core runtime
               FastAPI                 REST API framework
               Uvicorn                 ASGI web server
               SQLAlchemy              ORM (Object-Relational Mapper)
               SQLite                  Embedded database (dev)

AI/ML          Ultralytics YOLOv8m     Object detection model
               PyTorch 2.x             Deep learning runtime
               OpenCV                  Image preprocessing

Data           20,000+ images          Training corpus
               3 damage classes        Crack, Pothole, Surface Erosion
               YOLO annotation format  Normalized bounding boxes
               70/20/10 split          Train/Val/Test partitioning

Deployment     localhost:8000          FastAPI backend
               localhost:5500          Static frontend server
```

---

## 9. Key Interview Talking Points

### Q: Why YOLO over other models?
- Single-stage detector → **< 120ms inference** on CPU
- Pre-trained on COCO (transfer learning) → needs less custom data
- Bounding box + class + confidence in one forward pass
- YOLOv8m: balance of accuracy (mAP) and speed

### Q: How does GPS + locality work?
1. Browser `navigator.geolocation` API requests permission
2. Returns raw latitude/longitude
3. **Reverse geocoding** via free OpenStreetMap Nominatim API
4. Returns human-readable: "Arera Colony, Bhopal, Madhya Pradesh"
5. Stored in DB, displayed in complaints table

### Q: How is the 70/20/10 split deterministic?
- Uses `MD5(filename) % 100` — same file always goes to same split
- No randomness, fully reproducible across runs
- Pre-labelled data from `img/data/images/train|valid|test` preserves original split

### Q: What is the severity scoring logic?
```python
confidence ≥ 0.80  →  HIGH    (immediate repair needed)
confidence ≥ 0.55  →  MEDIUM  (schedule within a week)
confidence < 0.55  →  LOW     (monitor, low priority)
```

### Q: What is the full inference latency?
```
Image upload     :  ~50ms  (network, local)
File save        :  ~5ms
YOLO inference   :  ~80ms  (CPU, YOLOv8m, 640×640)
DB write         :  ~5ms
Dataset copy     :  ~5ms
JSON response    :  ~5ms
──────────────────────────
Total            : ~150ms end-to-end
```

### Q: What datasets were used?
| Source                  | Images | Annotation |
|-------------------------|--------|------------|
| Roboflow (Data Y12)     | 1,071  | YOLO .txt  |
| img/data/ (road issues) | 8,394  | YOLO .txt  |
| Kaggle (potholes etc.)  | 1,346  | YOLO .txt  |
| img/ (classified)       | ~9,000 | AI-labelled|
| **Total classified**    | **~20,000** | **3 classes** |

---

*Generated: Smart Civic AI v1.0 — VIT Bhopal University, 2026*
