from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import Base, engine
from models.user import User
from models.complaint import Complaint

from routes.auth import router as auth_router
from routes.complaint import router as complaint_router


app = FastAPI(
    title="Smart Civic AI",
    description="AI Powered Civic Infrastructure Monitoring System",
    version="1.0.0"
)



app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        # Local development
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        # Vercel production & preview deployments
        "https://smart-civic-ai.vercel.app",
        "https://ai-powered-infrstructure-monitoring-system.vercel.app",
        # Allow all vercel preview URLs for this project
        "https://*.vercel.app",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================
# DATABASE
# ============================================

Base.metadata.create_all(bind=engine)


# ============================================
# UPLOADED IMAGES
# ============================================

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)


# ============================================
# ROUTES
# ============================================

app.include_router(auth_router)
app.include_router(complaint_router)


# ============================================
# HOME
# ============================================

@app.get("/")
def home():
    return {
        "message": "Welcome to Smart Civic AI Backend",
        "status": "Running"
    }