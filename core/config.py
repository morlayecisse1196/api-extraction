"""
core/config.py
───────────────
Configuration centralisee via pydantic-settings.
Toutes les valeurs sont surchargeables par variables d'environnement.
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):

    # ── Chemins des modeles ───────────────────────────────────────────────────
    YOLO_MODEL_PATH:      str = "models/best.pt"
    COIN_MODEL_PATH:      str = "models/best_coin.pt"
    TROCR_MODEL_PATH:     str = "models/mon_modele_trocr_final_2"
    LAYOUTLM_MODEL_PATH:  str = "models/layoutlmv3_finetuned"
    GOOGLE_VISION_CREDENTIALS_PATH: str = "fichier-cles.json"
    OCR_ENGINE_DEFAULT: str = "trocr"

    # ── Parametres YOLO ───────────────────────────────────────────────────────
    YOLO_CONFIDENCE:  float = 0.25
    YOLO_IMGSZ:       int   = 1024

    # Redressement de page via detection des 4 coins
    ENABLE_PAGE_RECTIFICATION: bool = True
    COIN_CONFIDENCE:          float = 0.4 # A ajuster selon les resultats sur les coins (ex: 0.25 pour le modele de test)
    COIN_CONFIDENCE_PADDED:   float = 0.25 # A ajuster selon les resultats sur les coins (ex: 0.15 pour le modele de test)
    RECTIFY_PADDING_SIZE:     int   = 150
    RECTIFY_FINAL_WIDTH:      int   = 800
    RECTIFY_FINAL_HEIGHT:     int   = 1100
    RECTIFY_MIN_DOC_AREA_RATIO: float = 0.20
    RECTIFY_MAX_DOC_AREA_RATIO: float = 0.98
    RECTIFY_MIN_SIDE_PX:        int   = 120
    RECTIFY_MAX_ASPECT_RATIO:   float = 2.2
    RECTIFY_MIN_ASPECT_RATIO:   float = 0.45

    # Classes YOLO
    CLASS_TEXT_LINE:        int = 3
    CLASS_CHECKBOX_CHECKED: int = 0
    CLASS_CHECKBOX_EMPTY:   int = 1

    # Filtrage anti-bruit des detections YOLO (avant TrOCR)
    TEXT_LINE_MIN_CONF:      float = 0.35
    CHECKBOX_MIN_CONF:       float = 0.35
    TEXT_MIN_WIDTH_PX:       int   = 20
    TEXT_MIN_HEIGHT_PX:      int   = 10
    TEXT_MIN_AREA_PX2:       int   = 300
    TEXT_MAX_ASPECT_RATIO:   float = 35.0
    TEXT_REGION_X_MIN:       float = 0.02
    TEXT_REGION_X_MAX:       float = 0.98
    TEXT_REGION_Y_MIN:       float = 0.02
    TEXT_REGION_Y_MAX:       float = 0.98
    TEXT_MAX_WIDTH_RATIO:    float = 0.75
    MAX_TROCR_ZONES:         int   = 40
    MAP_CHECKBOX_EMPTY_TO_NON: bool = True

    # Optimisation TrOCR
    TROCR_MAX_NEW_TOKENS:    int   = 32
    TROCR_BATCH_SIZE:        int   = 8
    TROCR_MAX_TIME_S:        float = 25.0
    TROCR_MAX_CROP_PIXELS:   int   = 262144
    TROCR_REFERENCE_OCR_S:    float = 45.0

    # ── Parametres LayoutLMv3 ─────────────────────────────────────────────────
    LAYOUTLM_MAX_LENGTH: int = 512

    # ── API ───────────────────────────────────────────────────────────────────
    MAX_IMAGE_SIZE_MB: int   = 20
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png", "tiff", "bmp"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
