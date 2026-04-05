"""
models_loader/loader.py
────────────────────────
Charge les 3 modeles (YOLO, TrOCR, LayoutLMv3) une seule fois
au demarrage de l'API et les expose via ModelRegistry.
"""

import torch
from ultralytics import YOLO
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)
import json
import os
from pathlib import Path

from core.config import settings
from core.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """
    Conteneur pour les 3 modeles.
    Injecte dans app.state au demarrage via lifespan.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device : {self.device}")

        self.root_dir = Path(__file__).resolve().parents[1]

        self.coin:            YOLO                            = None
        self.yolo:            YOLO                            = None
        self.trocr_processor: TrOCRProcessor                 = None
        self.trocr_model:     VisionEncoderDecoderModel       = None
        self.lm_processor:    LayoutLMv3Processor             = None
        self.lm_model:        LayoutLMv3ForTokenClassification = None
        self.google_vision_client = None
        self.id2label:        dict                            = {}
        self.label2id:        dict                            = {}

    def load_all(self):
        self._load_google_vision()
        self._load_coin()
        self._load_yolo()
        self._load_trocr()
        self._load_layoutlm()

    # ── Google Vision OCR ───────────────────────────────────────────────────

    def _load_google_vision(self):
        path = self.root_dir / settings.GOOGLE_VISION_CREDENTIALS_PATH
        if not path.exists():
            logger.warning(f"Google Vision credentials introuvables : {path}")
            self.google_vision_client = None
            return

        try:
            from google.cloud import vision
        except Exception as e:
            logger.warning(f"Google Vision indisponible (package manquant) : {e}")
            self.google_vision_client = None
            return

        try:
            logger.info(f"Chargement Google Vision : {path}")
            self.google_vision_client = vision.ImageAnnotatorClient.from_service_account_json(str(path))
            logger.info("Google Vision charge")
        except Exception as e:
            logger.warning(f"Google Vision non initialise : {e}")
            self.google_vision_client = None

    # ── COINS (redressement page) ───────────────────────────────────────────

    def _load_coin(self):
        path = settings.COIN_MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"Coin model introuvable : {path} (redressement desactive)")
            self.coin = None
            return
        logger.info(f"Chargement Coin YOLO : {path}")
        self.coin = YOLO(path)
        logger.info("Coin YOLO charge")

    # ── YOLO ──────────────────────────────────────────────────────────────────

    def _load_yolo(self):
        path = settings.YOLO_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"YOLO model introuvable : {path}")
        logger.info(f"Chargement YOLO : {path}")
        self.yolo = YOLO(path)
        logger.info("YOLO charge")

    # ── TrOCR ─────────────────────────────────────────────────────────────────

    def _load_trocr(self):
        path = settings.TROCR_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"TrOCR model introuvable : {path}")
        logger.info(f"Chargement TrOCR : {path}")
        self.trocr_processor = TrOCRProcessor.from_pretrained(path)
        self.trocr_model     = VisionEncoderDecoderModel.from_pretrained(path)
        self.trocr_model.to(self.device)
        self.trocr_model.eval()
        logger.info("TrOCR charge")

    # ── LayoutLMv3 ────────────────────────────────────────────────────────────

    def _load_layoutlm(self):
        path = settings.LAYOUTLM_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"LayoutLMv3 model introuvable : {path}")
        logger.info(f"Chargement LayoutLMv3 : {path}")

        # Labels NER
        label_cfg_path = os.path.join(path, "label_config.json")
        with open(label_cfg_path) as f:
            cfg = json.load(f)
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.label2id = cfg["label2id"]

        self.lm_processor = LayoutLMv3Processor.from_pretrained(path, apply_ocr=False)
        self.lm_model     = LayoutLMv3ForTokenClassification.from_pretrained(path)
        self.lm_model.to(self.device)
        self.lm_model.eval()
        logger.info("LayoutLMv3 charge")
