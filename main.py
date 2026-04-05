"""
main.py
────────
Point d'entree de l'API FastAPI.
Les modeles sont charges UNE SEULE FOIS au demarrage (lifespan).
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import get_logger
from models_loader.loader import ModelRegistry
from routers import extract

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Charge tous les modeles au demarrage, les libere a l'arret.
    Pattern recommande FastAPI pour les ressources lourdes.
    """
    logger.info("Chargement des modeles...")
    registry = ModelRegistry()
    registry.load_all()
    app.state.models = registry
    logger.info("Tous les modeles sont prets.")
    yield
    logger.info("Arret de l'API — liberation des modeles.")
    del app.state.models


app = FastAPI(
    title="Autopsie Verbale — API d'extraction",
    description="Extrait les donnees structurees d'un formulaire d'autopsie verbale via YOLO + OCR hybride (Google Vision/TrOCR) + LayoutLMv3.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router, prefix="/api/v1", tags=["Extraction"])


@app.get("/health", tags=["Sante"])
async def health():
    return {"status": "ok", "version": "1.0.0"}
