<div align="center">

# 🚦 Smart Civic AI
### AI-Powered Civic Infrastructure Monitoring & Complaint Management System

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![VIT Bhopal](https://img.shields.io/badge/VIT-Bhopal%20University-blue?style=for-the-badge)](https://vitbhopal.ac.in)

**Transforming unstructured citizen photos into actionable, geo-located, severity-ranked civic work orders using computer vision.**

[Live Demo](https://smart-civic-ai.vercel.app) · [API Docs](http://localhost:8000/docs) · [Report Bug](https://github.com/BugnBuild/AI-powered-Infrstructure-monitoring-system/issues)

</div>

---

## 📸 Screenshots

| Dashboard (Dark Theme) | Report Modal with GPS | AI Detection Result |
|---|---|---|
| ![Dashboard](docs/dashboard.png) | ![Modal](docs/modal.png) | ![Result](docs/result.png) |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    SMART CIVIC AI PLATFORM                    │
├───────────────────┬──────────────────────┬───────────────────┤
│   FRONTEND        │      BACKEND         │    AI / DATA      │
│  HTML5 + CSS3     │  FastAPI + Uvicorn   │  YOLOv8m + SQLite │
│  JavaScript ES6   │  SQLAlchemy ORM      │  20,000+ images   │
│  Vercel (CDN)     │  Port 8000           │  3 damage classes │
└───────────────────┴──────────────────────┴───────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **AI Damage Detection** | YOLOv8m classifies Crack / Pothole / Surface Erosion in < 120ms |
| 📍 **GPS + Reverse Geocoding** | Auto-detects location and converts to human-readable locality using OpenStreetMap |
| 🗄️ **Complaint Dashboard** | Real-time table with severity badges, location, and status tracking |
| 📊 **Auto Dataset Builder** | Every submitted image is auto-classified and saved to train/val/test (70/20/10) |
| 🌙 **Dark Theme UI** | Modern glassmorphism design with animated AI detection mockup |
| 📱 **Responsive** | Works on desktop, tablet, and mobile |

---

## 🗂️ Project Structure

```
AI-Powered-Infrastructure-Monitoring-System/
│
├── frontend/                    # Static web app (Vercel)
│   ├── index.html               # Dashboard + report modal
│   ├── style.css                # Dark theme design system
│   └── script.js                # GPS, API calls, UI logic
│
├── backend/                     # FastAPI server
│   ├── app.py                   # App entry point + CORS
│   ├── database.py              # SQLAlchemy + SQLite setup
│   ├── requirements.txt         # Python dependencies
│   ├── config.py                # Environment config
│   ├── models/
│   │   ├── user.py              # User DB model
│   │   └── complaint.py         # Complaint DB model
│   ├── routes/
│   │   ├── auth.py              # Register / Login
│   │   └── complaint.py        # Submit + list complaints
│   ├── services/
│   │   ├── ai_service.py        # YOLOv8 inference wrapper
│   │   ├── auth_services.py     # JWT auth helpers
│   │   └── complaint_services.py
│   └── uploads/                 # Submitted images stored here
│
├── ai/
│   ├── scripts/
│   │   ├── 1_download_kaggle_datasets.py
│   │   ├── 2_merge_and_convert.py
│   │   ├── 3_split_dataset.py
│   │   ├── 4_train_yolo.py      # YOLOv8m training pipeline
│   │   ├── 5_evaluate_and_export.py
│   │   └── classify_all_images.py  # Auto-classify 20k images
│   └── dataset/
│       └── Data Y12 Final/      # Roboflow dataset (YOLO format)
│
├── DATA_FLOW_STRUCTURE.md       # Interview data flow document
├── README.md                    # This file
└── vercel.json                  # Vercel deployment config
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/BugnBuild/AI-powered-Infrstructure-monitoring-system.git
cd AI-powered-Infrstructure-monitoring-system
```

### 2. Set up the backend
```bash
cd backend
pip install -r requirements.txt
```

> **Note:** The trained model weights (`best.pt`) are not included in the repo due to file size.
> Either train your own (see [Training](#-model-training)) or download from [Releases](https://github.com/BugnBuild/AI-powered-Infrstructure-monitoring-system/releases).

### 3. Run the backend
```bash
# From the backend/ directory
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run the frontend
```bash
# From the frontend/ directory
python -m http.server 5500
# Open http://localhost:5500
```

---

## 🤖 Model Training

The YOLOv8m model is trained on **20,000+ road damage images** across 3 classes:

| Class | Description | Training Images |
|---|---|---|
| `Crack` | Longitudinal, transverse & alligator cracks | ~4,500 |
| `Pothole` | Asphalt voids and deep impact craters | ~9,000 |
| `Surface Erosion` | Surface wear, abrasion, raveling | ~6,500 |

### Run the full training pipeline:
```bash
cd ai/scripts

# Step 1: Download datasets from Kaggle
python 1_download_kaggle_datasets.py

# Step 2: Merge & convert annotations to YOLO format
python 2_merge_and_convert.py

# Step 3: Split into train/val/test (70/20/10)
python 3_split_dataset.py

# Step 4: Train YOLOv8m (100 epochs, early stopping)
python 4_train_yolo.py

# Step 5: Evaluate on test set + export ONNX
python 5_evaluate_and_export.py
```

### Training Config
```python
model   = YOLOv8m          # 25.9M parameters
epochs  = 100              # early stop patience=20
imgsz   = 640              # input resolution
batch   = auto             # fills 60% GPU VRAM
optimizer = AdamW          # with cosine LR schedule
```

---

## 📡 API Reference

```
POST   /complaints/submit    Upload image + GPS → AI analysis + save complaint
GET    /complaints/          List all complaints (JSON)
GET    /complaints/{id}      Single complaint detail
GET    /uploads/{filename}   Serve uploaded image
POST   /auth/register        Create user account
POST   /auth/login           Login, returns JWT token
GET    /                     Health check
GET    /docs                 Interactive Swagger UI
```

### Example Request
```bash
curl -X POST http://localhost:8000/complaints/submit \
  -F "file=@road_damage.jpg" \
  -F "user_id=1" \
  -F "latitude=23.2599" \
  -F "longitude=77.4126" \
  -F "locality=Arera Colony, Bhopal"
```

### Example Response
```json
{
  "complaint_id": 42,
  "analysis": {
    "damage_type": "Pothole",
    "confidence": 0.942,
    "severity": "HIGH",
    "bbox": [120, 85, 380, 290]
  },
  "location": {
    "latitude": 23.2599,
    "longitude": 77.4126,
    "locality": "Arera Colony, Bhopal"
  },
  "status": "Submitted"
}
```

---

## 🌍 Deployment

### Frontend — Vercel (Static)
The `frontend/` directory is deployed as a static site on Vercel.

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy from frontend/ directory
cd frontend
vercel --prod
```

Or connect the GitHub repo to Vercel — it auto-deploys on every push.

### Backend — Railway / Render / VPS
```bash
# Environment variables needed:
DATABASE_URL=sqlite:///./smart_civic.db
SECRET_KEY=your-secret-key
MODEL_PATH=./models/best.pt
```

> **Important:** After deploying the backend, update `API_URL` in `frontend/script.js` to your backend's public URL.

---

## 🧪 Data Flow

See [DATA_FLOW_STRUCTURE.md](DATA_FLOW_STRUCTURE.md) for the complete step-by-step data flow including:
- GPS permission + reverse geocoding pipeline
- AI inference pipeline (YOLO → severity matrix)
- Database schema
- Train/val/test split logic (deterministic MD5 hash)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, JavaScript ES6 |
| Backend | Python 3.13, FastAPI, Uvicorn |
| Database | SQLAlchemy ORM, SQLite |
| AI/ML | YOLOv8m (Ultralytics), PyTorch, OpenCV |
| GPS | Browser Geolocation API + OpenStreetMap Nominatim |
| Hosting | Vercel (frontend), Railway/Render (backend) |

---

## 👥 Team

| Name | Roll No | Role |
|---|---|---|
| **Raghwendra Singh** | 23BET10006 | Full Stack + AI/ML |
| **Krish Sachan** | 23BET10033 | Backend + Database |
| **Abhishek Singh** | 23BET10048 | Frontend + Dataset |

**Institution:** VIT Bhopal University, Department of CSE  
**Project Supervisor:** Dr. P R Bhuvaneswari

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Made with ❤️ at VIT Bhopal University · 2026
</div>
