"""
routers/extract.py
───────────────────
Endpoint principal :  POST /api/v1/extract
.......

Accepte :
  - Une image (UploadFile)
  - Optionnel : page_number (int) pour adapter le mapping de position

Retourne :
  {
    "status": "success",
    "page":   1,
    "data": {
      "enqueteur":   "Tapo Nickhon",
      "date_visite": "01/03/2026",
      ...
      "_meta": { ... }
    },
    "raw_predictions": [  # optionnel si debug=true
      {"token": "...", "bbox": [...], "ner_label": "..."},
      ...
    ]
  }
"""

import io
import time
from typing import Annotated
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

from core.config import settings
from core.logger import get_logger
from services.pipeline import run_pipeline, run_pipeline_with_profile
from services.mapper import map_predictions, map_predictions_page_labels

logger = get_logger(__name__)
router = APIRouter()


def _as_score(conf_map: dict, key: str) -> float:
    try:
        return float(conf_map.get(key, {}).get("score", 0.0))
    except Exception:
        return 0.0


def _build_top_predictions(predictions: list[dict], limit: int = 12) -> list[dict]:
    scored = []
    for pred in predictions:
        yolo_conf = float(pred.get("yolo_conf", 0.0))
        ner_conf = float(pred.get("ner_conf", 0.0))
        score = (0.4 * yolo_conf) + (0.6 * ner_conf)
        scored.append({
            "token": pred.get("token", ""),
            "ner_label": pred.get("ner_label", "O"),
            "bbox": pred.get("bbox", []),
            "bbox_norm": pred.get("bbox_norm", []),
            "yolo_conf": round(yolo_conf, 4),
            "ner_conf": round(ner_conf, 4),
            "score": round(score, 4),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max(1, limit)]


def _validate_image(file: UploadFile) -> None:
    """Valide l'extension et la taille du fichier uploade."""
    ext = file.filename.rsplit(".", 1)[-1].lower() if file.filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Format non supporte : .{ext}. Acceptes : {settings.ALLOWED_EXTENSIONS}"
        )


@router.post(
    "/extract",
    summary="Extraire les donnees d'un formulaire d'autopsie verbale",
    response_description="JSON structure avec les vrais labels du formulaire",
    responses={400: {"description": "Image illisible"}, 415: {"description": "Format non supporte"}},
)
async def extract(
    request:     Request,
    file:        Annotated[UploadFile, File(..., description="Image du formulaire (jpg, png, tiff)")],
    page_number: Annotated[int, Query(ge=1, le=4, description="Numero de page du formulaire (1-4)")] = 1,
    ocr_engine:  Annotated[str, Query(description="Moteur OCR: trocr ou google")] = settings.OCR_ENGINE_DEFAULT,
    debug:       Annotated[bool, Query(description="Inclure les predictions brutes dans la reponse")] = False,
):
    """
    Pipeline complet d'extraction :

    1. Validation de l'image
    2. YOLO OBB → detection des zones manuscrites
    3. TrOCR → lecture du texte dans chaque zone
    4. LayoutLMv3 → prediction des labels NER
    5. Mapper → correspondance avec les vrais labels du formulaire
    """
    _validate_image(file)

    # Lecture image
    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossible de lire l'image : {e}")

    logger.info(f"Extraction : {file.filename} | {pil_img.size} | page {page_number}")
    logger.info(f"OCR engine demande={ocr_engine} debug={debug}")
    t0 = time.perf_counter()

    # Recuperation des modeles depuis app.state
    registry = request.app.state.models

    # Pipeline
    pipeline_profile = None
    if debug:
        predictions, pipeline_profile = run_pipeline_with_profile(pil_img, registry, ocr_engine=ocr_engine)
    else:
        predictions = run_pipeline(pil_img, registry, ocr_engine=ocr_engine)

    ocr_used = ocr_engine
    fallback_reason = ""
    if pipeline_profile:
        ocr_used = (pipeline_profile or {}).get("ocr", {}).get("used_engine", ocr_engine)
        fallback_reason = (pipeline_profile or {}).get("ocr", {}).get("fallback_reason", "")
    logger.info(f"OCR engine utilise={ocr_used} fallback={fallback_reason or 'none'}")

    if not predictions:
        return JSONResponse(
            status_code=200,
            content={
                "status":  "empty",
                "page":    page_number,
                "message": "Aucune zone detectee dans l'image.",
                "data":    {},
            }
        )

    # Mapping vers sortie stricte par page
    structured = map_predictions_page_labels(predictions, page_number=page_number)

    elapsed = round(time.perf_counter() - t0, 3)
    logger.info(f"Extraction terminee en {elapsed}s — page {page_number}")

    response = structured

    if debug:
        response["raw_predictions"] = predictions
        rect_info = (pipeline_profile or {}).get("rectification", {})
        response["pipeline_debug"] = {
            "ocr": (pipeline_profile or {}).get("ocr", {}),
            "rectification": (pipeline_profile or {}).get("rectification", {}),
            "counts": (pipeline_profile or {}).get("counts", {}),
            "filters": (pipeline_profile or {}).get("filters", {}),
            "timings_s": (pipeline_profile or {}).get("timings_s", {}),
            "boxes": {
                "page_before_rectification": rect_info.get("source_quad", []),
                "page_after_rectification": rect_info.get("destination_quad", []),
            },
            "top_predictions": _build_top_predictions(predictions),
        }

    return JSONResponse(content=response)


@router.post(
    "/extract/batch",
    summary="Extraire les donnees de plusieurs pages en une seule requete",
    responses={400: {"description": "Nombre de pages invalide"}, 415: {"description": "Format non supporte"}},
)
async def extract_batch(
    request: Request,
    files:   Annotated[list[UploadFile], File(..., description="Images des pages du formulaire")],
    ocr_engine: Annotated[str, Query(description="Moteur OCR: trocr ou google")] = settings.OCR_ENGINE_DEFAULT,
    debug:   Annotated[bool, Query()] = False,
):
    """
    Traite plusieurs pages d'un meme formulaire.
    Les resultats sont fusionnes en un seul JSON.
    """
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 pages par requete.")

    registry = request.app.state.models
    all_data = {}
    all_conf = {}
    pages_meta = []
    batch_dropped_low_conf = 0
    t0 = time.perf_counter()

    for idx, file in enumerate(files, start=1):
        _validate_image(file)
        try:
            contents = await file.read()
            pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            logger.warning(f"Page {idx} ignoree : {e}")
            continue

        logger.info(f"Batch page {idx}/{len(files)} : {file.filename}")
        predictions = run_pipeline(pil_img, registry, ocr_engine=ocr_engine)
        structured  = map_predictions(predictions, page_number=idx)

        conf_map = structured.pop("_confidence", {})
        meta = structured.pop("_meta", {})
        pages_meta.append({"page": idx, "filename": file.filename, **meta})

        # Fusion robuste : conserver la valeur la plus confiante par champ
        for key, value in structured.items():
            incoming_score = _as_score(conf_map, key)
            if key not in all_data:
                all_data[key] = value
                if key in conf_map:
                    all_conf[key] = conf_map[key]
                    all_conf[key]["page"] = idx
            else:
                existing_score = _as_score(all_conf, key)
                if incoming_score >= existing_score:
                    all_data[key] = value
                    if key in conf_map:
                        all_conf[key] = conf_map[key]
                        all_conf[key]["page"] = idx
                    batch_dropped_low_conf += 1
                else:
                    batch_dropped_low_conf += 1

    if all_conf:
        all_data["_confidence"] = all_conf
    all_data["_meta"] = {
        "batch_dropped_low_conf": batch_dropped_low_conf,
        "pages_processed": len(pages_meta),
    }

    elapsed = round(time.perf_counter() - t0, 3)
    return JSONResponse(content={
        "status":            "success",
        "pages_processed":   len(pages_meta),
        "processing_time_s": elapsed,
        "pages_meta":        pages_meta,
        "data":              all_data,
    })
