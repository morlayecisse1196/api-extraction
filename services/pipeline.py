"""
services/pipeline.py
──────────────────────
Orchestrateur principal :
  1. YOLO OBB  → detecte les zones manuscrites
  2. TrOCR     → lit le texte dans chaque zone
  3. LayoutLMv3 → predit les labels NER
  4. Retourne liste de {token, bbox, ner_label}
"""

import cv2
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from time import perf_counter

from core.config import settings
from core.logger import get_logger
from models_loader.loader import ModelRegistry

logger = get_logger(__name__)

OCR_EMPTY_TOKEN = "[VIDE]"
OCR_TIMEOUT_TOKEN = "[TIMEOUT]"


# ── Helpers geometrie ─────────────────────────────────────────────────────────

def get_rotated_crop(img: np.ndarray, points: list) -> np.ndarray:
    """Redresse une zone OBB par warpPerspective."""
    pts  = np.array(points, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    max_width  = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
    max_height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
    if max_width < 1 or max_height < 1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    dst = np.array([[0,0],[max_width-1,0],[max_width-1,max_height-1],[0,max_height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))


def obb_to_aabb(points: list) -> list[int]:
    """4 points OBB → [x1, y1, x2, y2] rectangle droit."""
    pts = np.array(points).reshape(4, 2)
    return [int(np.min(pts[:,0])), int(np.min(pts[:,1])),
            int(np.max(pts[:,0])), int(np.max(pts[:,1]))]


def normalize_bbox(x1, y1, x2, y2, w, h) -> list[int]:
    """Pixels → [0, 1000] pour LayoutLMv3."""
    return [
        round((x1 / w) * 1000), round((y1 / h) * 1000),
        round((x2 / w) * 1000), round((y2 / h) * 1000),
    ]


def _vertices_to_poly(vertices) -> np.ndarray:
    pts = []
    for vertex in vertices or []:
        x = int(getattr(vertex, "x", 0) or 0)
        y = int(getattr(vertex, "y", 0) or 0)
        pts.append([x, y])

    if len(pts) < 4:
        while len(pts) < 4:
            pts.append([0, 0])

    return np.array(pts[:4], dtype=np.float32)


def _bbox_from_poly(poly: np.ndarray) -> list[int]:
    if poly.size == 0:
        return [0, 0, 0, 0]
    xs = poly[:, 0]
    ys = poly[:, 1]
    return [int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))]


def _polygon_center(poly: np.ndarray) -> tuple[float, float]:
    if poly.size == 0:
        return 0.0, 0.0
    return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))


def _point_in_poly(point: tuple[float, float], poly: np.ndarray) -> bool:
    if poly.size == 0:
        return False
    return cv2.pointPolygonTest(poly.astype(np.float32), point, False) >= 0


def get_text_in_zone(yolo_aabb: list[int], google_tokens: list[dict]) -> list[dict]:
    """
    Retourne les tokens Google dont le centre (x,y) est dans l'AABB YOLO.
    Tri de lecture local : gauche -> droite.
    """
    if not yolo_aabb or len(yolo_aabb) != 4:
        return []

    x1, y1, x2, y2 = yolo_aabb
    matched: list[dict] = []

    for token in google_tokens:
        cx = float(token.get("center_x", 0.0))
        cy = float(token.get("center_y", 0.0))
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            matched.append(token)

    if not matched:
        return matched

    # Smart reading order local: groupe par lignes (Y proche), puis X croissant.
    heights = []
    for token in matched:
        bbox = token.get("bbox", [0, 0, 0, 0])
        if isinstance(bbox, list) and len(bbox) == 4:
            heights.append(max(1.0, float(bbox[3] - bbox[1])))

    h_avg = (sum(heights) / len(heights)) if heights else 12.0
    y_threshold = max(3.0, h_avg * 0.55)

    matched.sort(key=lambda item: (item.get("center_y", 0.0), item.get("center_x", 0.0)))
    line_buckets: list[dict] = []

    for token in matched:
        cy = float(token.get("center_y", 0.0))
        placed = False
        for bucket in line_buckets:
            if abs(cy - float(bucket["y_center"])) <= y_threshold:
                bucket["items"].append(token)
                ys = [float(it.get("center_y", 0.0)) for it in bucket["items"]]
                bucket["y_center"] = sum(ys) / max(1, len(ys))
                placed = True
                break
        if not placed:
            line_buckets.append({"y_center": cy, "items": [token]})

    ordered: list[dict] = []
    line_buckets.sort(key=lambda b: float(b["y_center"]))
    for bucket in line_buckets:
        row = sorted(bucket["items"], key=lambda it: float(it.get("center_x", 0.0)))
        ordered.extend(row)

    return ordered


def get_text_in_obb(obb_points: list, google_tokens: list[dict]) -> list[dict]:
    """
    Retourne les tokens Google dont le centre ou la majorite de la boite
    est a l'interieur du polygone OBB YOLO.
    """
    poly = np.array(obb_points, dtype=np.float32).reshape(-1, 2)
    matched = []

    for token in google_tokens:
        bbox = token.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = bbox
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0
        points = [
            (cx, cy),
            (float(x1), float(y1)),
            (float(x1), float(y2)),
            (float(x2), float(y1)),
            (float(x2), float(y2)),
        ]
        inside_count = sum(1 for pt in points if _point_in_poly(pt, poly))
        if inside_count >= 3:
            matched.append(token)

    matched.sort(key=lambda item: (item.get("center_y", 0.0), item.get("center_x", 0.0)))
    return matched


def _concat_google_tokens(tokens: list[dict]) -> str:
    if not tokens:
        return OCR_EMPTY_TOKEN
    text = " ".join(token.get("token", "").strip() for token in tokens if token.get("token", "").strip())
    return text if text.strip() else OCR_EMPTY_TOKEN


def run_google_vision_document_ocr(pil_img: Image.Image, registry: ModelRegistry) -> tuple[list[dict], dict]:
    """
    OCR Google Vision sur l'image complete.
    Retourne une liste de mots avec leurs boites.
    """
    client = getattr(registry, "google_vision_client", None)
    if client is None:
        return [], {"ok": False, "reason": "missing_client", "tokens": 0}

    try:
        from google.cloud import vision
    except Exception as e:
        return [], {"ok": False, "reason": f"package_missing:{e}", "tokens": 0}

    try:
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        image = vision.Image(content=buffer.getvalue())
        response = client.document_text_detection(image=image)

        if getattr(response, "error", None) and getattr(response.error, "message", ""):
            return [], {"ok": False, "reason": response.error.message, "tokens": 0}

        google_tokens: list[dict] = []
        full_text = getattr(response, "full_text_annotation", None)
        pages = getattr(full_text, "pages", []) if full_text else []

        for page_index, page in enumerate(pages):
            for block in getattr(page, "blocks", []) or []:
                for paragraph in getattr(block, "paragraphs", []) or []:
                    for word in getattr(paragraph, "words", []) or []:
                        word_text = "".join(symbol.text for symbol in getattr(word, "symbols", []) or []).strip()
                        if not word_text:
                            continue

                        poly = _vertices_to_poly(getattr(word.bounding_box, "vertices", []))
                        bbox = _bbox_from_poly(poly)
                        center_x, center_y = _polygon_center(poly)
                        google_tokens.append({
                            "token": word_text,
                            "bbox": bbox,
                            "bbox_norm": [],
                            "poly": poly.tolist(),
                            "center_x": center_x,
                            "center_y": center_y,
                            "confidence": float(getattr(word, "confidence", 0.0) or 0.0),
                            "page_index": page_index,
                        })

        logger.info(f"Google Vision : {len(google_tokens)} mots detectes")
        return google_tokens, {
            "ok": True,
            "reason": "ok",
            "tokens": len(google_tokens),
        }
    except Exception as e:
        logger.warning(f"Google Vision OCR erreur : {e}")
        return [], {"ok": False, "reason": str(e), "tokens": 0}


def _validate_rectification_geometry(rect: np.ndarray, img_w: int, img_h: int) -> tuple[bool, str]:
    """Valide que le quadrilatere detecte ressemble a une page exploitable."""
    if rect.shape != (4, 2):
        return False, "invalid_shape"

    # Surface du quadrilatere detecte vs surface image
    area = abs(float(cv2.contourArea(rect.astype(np.float32))))
    img_area = float(img_w * img_h)
    if img_area <= 0:
        return False, "invalid_image_area"
    area_ratio = area / img_area
    if area_ratio < settings.RECTIFY_MIN_DOC_AREA_RATIO or area_ratio > settings.RECTIFY_MAX_DOC_AREA_RATIO:
        return False, "invalid_area_ratio"

    tl, tr, br, bl = rect
    top = float(np.linalg.norm(tr - tl))
    right = float(np.linalg.norm(br - tr))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))

    if min(top, right, bottom, left) < float(settings.RECTIFY_MIN_SIDE_PX):
        return False, "side_too_small"

    width_avg = (top + bottom) / 2.0
    height_avg = (left + right) / 2.0
    if width_avg <= 1.0 or height_avg <= 1.0:
        return False, "invalid_dimensions"

    aspect = width_avg / height_avg
    if aspect < settings.RECTIFY_MIN_ASPECT_RATIO or aspect > settings.RECTIFY_MAX_ASPECT_RATIO:
        return False, "invalid_aspect_ratio"

    return True, "ok"


def _is_text_detection_valid(det: dict, img_w: int, img_h: int) -> tuple[bool, str]:
    """Filtre les detections bruites pour les zones texte."""
    x1, y1, x2, y2 = det["aabb"]
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    conf = float(det["conf"])

    if conf < settings.TEXT_LINE_MIN_CONF:
        return False, "low_confidence"
    if w < settings.TEXT_MIN_WIDTH_PX:
        return False, "too_narrow"
    if h < settings.TEXT_MIN_HEIGHT_PX:
        return False, "too_short"
    if area < settings.TEXT_MIN_AREA_PX2:
        return False, "too_small_area"

    width_ratio = w / max(1, img_w)
    if width_ratio > settings.TEXT_MAX_WIDTH_RATIO:
        return False, "too_wide_line"

    ratio = (w / h) if h > 0 else 999.0
    if ratio > settings.TEXT_MAX_ASPECT_RATIO:
        return False, "too_thin_ratio"

    cx = ((x1 + x2) / 2.0) / max(1, img_w)
    cy = ((y1 + y2) / 2.0) / max(1, img_h)
    if not (settings.TEXT_REGION_X_MIN <= cx <= settings.TEXT_REGION_X_MAX):
        return False, "out_of_useful_region_x"
    if not (settings.TEXT_REGION_Y_MIN <= cy <= settings.TEXT_REGION_Y_MAX):
        return False, "out_of_useful_region_y"

    return True, "ok"


def _prepare_trocr_crop(crop_pil: Image.Image) -> Image.Image:
    """Redimensionne les crops trop grands pour limiter le cout TrOCR."""
    w, h = crop_pil.size
    pixels = w * h
    max_pixels = max(1, settings.TROCR_MAX_CROP_PIXELS)
    if pixels <= max_pixels:
        return crop_pil

    scale = (max_pixels / float(pixels)) ** 0.5
    new_w = max(8, int(w * scale))
    new_h = max(8, int(h * scale))
    return crop_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)


def rectify_page_with_coin(pil_img: Image.Image, registry: ModelRegistry) -> Image.Image:
    """
    Redresse la page avec le modele de coins (4 coins attendus).
    Fallback : retourne l'image d'origine si detection insuffisante.
    """
    rectified, _ = rectify_page_with_coin_profile(pil_img, registry)
    return rectified


def rectify_page_with_coin_profile(pil_img: Image.Image, registry: ModelRegistry) -> tuple[Image.Image, dict]:
    """
    Variante profilee du redressement qui retourne aussi les metadonnees.
    """
    if not settings.ENABLE_PAGE_RECTIFICATION or registry.coin is None:
        return pil_img, {
            "applied": False,
            "reason": "disabled_or_missing_model",
            "coins_detected": 0,
            "used_padding": False,
            "source_quad": [],
            "destination_quad": [],
        }

    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    def _predict_boxes(img_source, conf: float):
        results = registry.coin.predict(source=img_source, conf=conf, verbose=False)
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 4)), np.empty((0,))
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        return boxes, confs

    boxes, confs = _predict_boxes(img_cv, settings.COIN_CONFIDENCE)
    used_padding = False
    pad_size = settings.RECTIFY_PADDING_SIZE

    if len(boxes) < 4:
        img_padded = cv2.copyMakeBorder(
            img_cv,
            pad_size,
            pad_size,
            pad_size,
            pad_size,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        boxes, confs = _predict_boxes(img_padded, settings.COIN_CONFIDENCE_PADDED)
        used_padding = True

    if len(boxes) < 4:
        logger.warning(f"Redressement ignore : {len(boxes)} coin(s) detecte(s)")
        return pil_img, {
            "applied": False,
            "reason": "not_enough_coins",
            "coins_detected": int(len(boxes)),
            "used_padding": used_padding,
            "source_quad": [],
            "destination_quad": [],
        }

    # Garde les 4 detections les plus confiantes
    top_idx = np.argsort(-confs)[:4]
    boxes = boxes[top_idx]

    pts = []
    for box in boxes:
        x_center = float((box[0] + box[2]) / 2.0)
        y_center = float((box[1] + box[3]) / 2.0)
        if used_padding:
            x_center -= pad_size
            y_center -= pad_size
        x_center = max(0.0, min(x_center, float(w - 1)))
        y_center = max(0.0, min(y_center, float(h - 1)))
        pts.append([x_center, y_center])

    pts_source = np.array(pts, dtype="float32")

    rect = np.zeros((4, 2), dtype="float32")
    s = pts_source.sum(axis=1)
    rect[0] = pts_source[np.argmin(s)]
    rect[2] = pts_source[np.argmax(s)]
    diff = np.diff(pts_source, axis=1)
    rect[1] = pts_source[np.argmin(diff)]
    rect[3] = pts_source[np.argmax(diff)]

    is_valid_geom, geom_reason = _validate_rectification_geometry(rect, w, h)
    if not is_valid_geom:
        logger.warning(f"Redressement ignore : geometrie invalide ({geom_reason})")
        return pil_img, {
            "applied": False,
            "reason": geom_reason,
            "coins_detected": 4,
            "used_padding": used_padding,
            "source_quad": rect.astype(int).tolist(),
            "destination_quad": [],
        }

    final_w = settings.RECTIFY_FINAL_WIDTH
    final_h = settings.RECTIFY_FINAL_HEIGHT
    pts_dest = np.array(
        [[0, 0], [final_w - 1, 0], [final_w - 1, final_h - 1], [0, final_h - 1]],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, pts_dest)
    rectified = cv2.warpPerspective(img_cv, matrix, (final_w, final_h))
    logger.info(f"Page redressee : {w}x{h} -> {final_w}x{final_h}")
    return Image.fromarray(cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)), {
        "applied": True,
        "reason": "ok",
        "coins_detected": 4,
        "used_padding": used_padding,
        "input_size": [int(w), int(h)],
        "output_size": [int(final_w), int(final_h)],
        "source_quad": rect.astype(int).tolist(),
        "destination_quad": pts_dest.astype(int).tolist(),
    }


# ── Etape 1 : YOLO ────────────────────────────────────────────────────────────

def run_yolo(pil_img: Image.Image, registry: ModelRegistry) -> list[dict]:
    """
    Detecte les zones OBB sur l'image.
    Retourne liste triee haut→bas de {cls_id, pts, aabb}.
    """
    results = registry.yolo.predict(
        source=pil_img,
        imgsz=settings.YOLO_IMGSZ,
        conf=settings.YOLO_CONFIDENCE,
        verbose=False,
    )
    r = results[0]
    if r.obb is None or len(r.obb) == 0:
        logger.warning("YOLO : aucune detection")
        return []

    detections = []
    for pts, cls, conf in zip(r.obb.xyxyxyxy.tolist(), r.obb.cls.tolist(), r.obb.conf.tolist()):
        aabb = obb_to_aabb(pts)
        x1, y1, x2, y2 = aabb
        y_mean = (y1 + y2) / 2.0
        x_mean = (x1 + x2) / 2.0
        height = max(1.0, float(y2 - y1))
        detections.append({
            "y_mean":  y_mean,
            "x_mean":  x_mean,
            "height":  height,
            "cls_id":  int(cls),
            "conf":    float(conf),
            "pts":     pts,
            "aabb":    aabb,
        })

    # Smart reading order: groupe par lignes (Y proche), puis X croissant.
    heights = [float(d.get("height", 1.0)) for d in detections]
    h_avg = sum(heights) / max(1, len(heights))
    y_threshold = max(4.0, h_avg / 2.0)

    detections.sort(key=lambda d: (d["y_mean"], d["aabb"][0]))
    line_buckets: list[dict] = []
    for det in detections:
        placed = False
        for bucket in line_buckets:
            if abs(det["y_mean"] - bucket["y_center"]) < y_threshold:
                bucket["items"].append(det)
                ys = [item["y_mean"] for item in bucket["items"]]
                bucket["y_center"] = sum(ys) / len(ys)
                placed = True
                break
        if not placed:
            line_buckets.append({"y_center": det["y_mean"], "items": [det]})

    ordered: list[dict] = []
    line_buckets.sort(key=lambda b: b["y_center"])
    for bucket in line_buckets:
        row = sorted(bucket["items"], key=lambda d: d["aabb"][0])
        ordered.extend(row)

    detections = ordered
    logger.info(f"YOLO : {len(detections)} zones detectees")
    return detections


# ── Etape 2 : TrOCR ───────────────────────────────────────────────────────────

def run_trocr_batch(crops_pil: list[Image.Image], registry: ModelRegistry) -> tuple[list[str], dict]:
    """
    Lit le texte d'une liste de crops PIL avec TrOCR en mode batch.
    Retourne (tokens, meta).
    """
    if not crops_pil:
        return [], {"processed": 0, "skipped_timeout": 0, "errors": 0}

    outputs: list[str] = []
    processed = 0
    skipped_timeout = 0
    errors = 0
    t0 = perf_counter()
    batch_size = max(1, settings.TROCR_BATCH_SIZE)

    for i in range(0, len(crops_pil), batch_size):
        elapsed = perf_counter() - t0
        if elapsed >= settings.TROCR_MAX_TIME_S:
            remaining = len(crops_pil) - i
            outputs.extend([OCR_TIMEOUT_TOKEN] * remaining)
            skipped_timeout += remaining
            break

        batch = [img.convert("RGB") for img in crops_pil[i : i + batch_size]]
        try:
            px = registry.trocr_processor(images=batch, return_tensors="pt", padding=True).pixel_values
            px = px.to(registry.device)

            with torch.no_grad():
                ids = registry.trocr_model.generate(
                    px,
                    max_new_tokens=settings.TROCR_MAX_NEW_TOKENS,
                )

            decoded = registry.trocr_processor.batch_decode(ids, skip_special_tokens=True)
            cleaned = [txt.strip() if txt and txt.strip() else OCR_EMPTY_TOKEN for txt in decoded]
            outputs.extend(cleaned)
            processed += len(batch)
        except Exception as e:
            logger.warning(f"TrOCR batch erreur : {e}")
            outputs.extend(["[ERREUR]"] * len(batch))
            errors += len(batch)

    return outputs, {"processed": processed, "skipped_timeout": skipped_timeout, "errors": errors}


# ── Etape 3 : LayoutLMv3 ─────────────────────────────────────────────────────

def run_layoutlm(
    pil_img:   Image.Image,
    tokens:    list[str],
    bboxes:    list[list[int]],
    registry:  ModelRegistry,
) -> tuple[list[str], list[float]]:
    """
    Predit les labels NER (format BIO) pour chaque token.
    Retourne (labels, confidences) de meme longueur que tokens.
    """
    if not tokens:
        return [], []

    try:
        enc = registry.lm_processor(
            pil_img,
            tokens,
            boxes=bboxes,
            truncation=True,
            padding="max_length",
            max_length=settings.LAYOUTLM_MAX_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = registry.lm_model(
                input_ids      = enc["input_ids"].to(registry.device),
                attention_mask = enc["attention_mask"].to(registry.device),
                bbox           = enc["bbox"].to(registry.device),
                pixel_values   = enc["pixel_values"].to(registry.device),
            ).logits

        probs = torch.softmax(logits, dim=-1)
        pred_conf = probs.max(dim=-1).values.squeeze().cpu().numpy()
        preds = logits.argmax(dim=-1).squeeze().cpu().numpy()
        word_ids = enc.word_ids(0)

        # Reconstruction : 1 prediction par mot (pas par sous-token WordPiece)
        token_labels = ["O"] * len(tokens)
        token_scores = [0.0] * len(tokens)
        seen = set()
        for tok_idx, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx in seen:
                continue
            if word_idx < len(tokens):
                token_labels[word_idx] = registry.id2label.get(int(preds[tok_idx]), "O")
                token_scores[word_idx] = float(pred_conf[tok_idx])
                seen.add(word_idx)

        return token_labels, token_scores

    except Exception as e:
        logger.error(f"LayoutLMv3 erreur : {e}")
        return ["O"] * len(tokens), [0.0] * len(tokens)


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_pipeline(pil_img: Image.Image, registry: ModelRegistry, ocr_engine: str = "trocr") -> list[dict]:
    """
    Pipeline principal (mode production).
    """
    results, _ = _run_pipeline_impl(pil_img, registry, collect_profile=False, ocr_engine=ocr_engine)
    return results


def run_pipeline_with_profile(
    pil_img: Image.Image,
    registry: ModelRegistry,
    ocr_engine: str = "trocr",
) -> tuple[list[dict], dict]:
    """
    Pipeline profile pour benchmark/diagnostic.
    Retourne (predictions, profile).
    """
    return _run_pipeline_impl(pil_img, registry, collect_profile=True, ocr_engine=ocr_engine)


def _run_pipeline_impl(
    pil_img: Image.Image,
    registry: ModelRegistry,
    collect_profile: bool,
    ocr_engine: str = "trocr",
) -> tuple[list[dict], dict]:
    """
    Pipeline complet sur une image PIL.

    Retourne liste de :
    {
        "token":      str,      # texte lu par TrOCR
        "bbox":       [x1,y1,x2,y2],  # pixels absolus
        "bbox_norm":  [0-1000] * 4,   # normalise pour LayoutLMv3
        "ner_label":  str,      # label predit (ex: "B-IDENTITE")
    }
    """
    t_pipeline = perf_counter()

    ocr_engine = (ocr_engine or settings.OCR_ENGINE_DEFAULT).strip().lower()
    if ocr_engine not in {"trocr", "google"}:
        ocr_engine = settings.OCR_ENGINE_DEFAULT

    profile = {
        "timings_s": {
            "rectification": 0.0,
            "ocr": 0.0,
            "yolo": 0.0,
            "trocr": 0.0,
            "layoutlm": 0.0,
            "assemble": 0.0,
        },
        "ocr": {
            "requested_engine": ocr_engine,
            "used_engine": "trocr",
            "fallback_reason": "",
            "google_tokens": 0,
            "google_matched_tokens": 0,
            "reference_trocr_s": float(settings.TROCR_REFERENCE_OCR_S),
            "gain_vs_trocr_s": 0.0,
        },
        "counts": {
            "detections": 0,
            "detections_filtered": 0,
            "tokens": 0,
            "trocr_calls": 0,
        },
        "filters": {
            "text_dropped_reasons": {},
            "trocr_skipped_by_limit": 0,
            "trocr_skipped_by_timeout": 0,
            "trocr_errors": 0,
        },
        "rectification": {
            "applied": False,
            "reason": "not_requested",
            "coins_detected": 0,
            "used_padding": False,
        },
    }

    # 0. Redressement page (coin model) avant detection des zones manuscrites
    t_rect = perf_counter()
    work_img, rect_meta = rectify_page_with_coin_profile(pil_img, registry)
    profile["timings_s"]["rectification"] = round(perf_counter() - t_rect, 6)
    profile["rectification"] = rect_meta

    img_w, img_h = work_img.size
    img_cv = cv2.cvtColor(np.array(work_img), cv2.COLOR_RGB2BGR)

    google_tokens: list[dict] = []
    google_ocr_elapsed = 0.0

    if ocr_engine == "google":
        t_ocr = perf_counter()
        google_tokens, google_ocr_meta = run_google_vision_document_ocr(work_img, registry)
        google_ocr_elapsed = perf_counter() - t_ocr
        profile["timings_s"]["ocr"] = round(google_ocr_elapsed, 6)
        profile["ocr"]["google_tokens"] = int(google_ocr_meta.get("tokens", 0))

        if google_ocr_meta.get("ok") and google_tokens:
            profile["ocr"]["used_engine"] = "google"
            gain = float(settings.TROCR_REFERENCE_OCR_S) - google_ocr_elapsed
            profile["ocr"]["gain_vs_trocr_s"] = round(gain, 6)
            gain_pct = (gain / float(settings.TROCR_REFERENCE_OCR_S) * 100.0) if settings.TROCR_REFERENCE_OCR_S > 0 else 0.0
            logger.info(
                f"OCR Google Vision: {google_ocr_elapsed:.2f}s vs TrOCR ref {settings.TROCR_REFERENCE_OCR_S:.2f}s "
                f"=> gain {gain:.2f}s ({gain_pct:.1f}%)"
            )
        else:
            profile["ocr"]["fallback_reason"] = str(google_ocr_meta.get("reason", "unknown"))
            logger.warning(
                f"Google Vision indisponible ou vide ({profile['ocr']['fallback_reason']}); bascule TrOCR"
            )

    # 1. Detection YOLO
    t_yolo = perf_counter()
    detections = run_yolo(work_img, registry)
    profile["timings_s"]["yolo"] = round(perf_counter() - t_yolo, 6)
    profile["counts"]["detections"] = len(detections)
    if not detections:
        return [], profile if collect_profile else {}

    # 2. Preparation des entrees texte + cases a cocher
    entries = []
    text_crops: list[Image.Image] = []
    trocr_calls_planned = 0

    for det in detections:
        cls_id = det["cls_id"]
        aabb   = det["aabb"]
        x1, y1, x2, y2 = aabb
        x1 = max(0, min(x1, img_w)); x2 = max(0, min(x2, img_w))
        y1 = max(0, min(y1, img_h)); y2 = max(0, min(y2, img_h))
        bbox = [x1, y1, x2, y2]
        norm_bb = normalize_bbox(x1, y1, x2, y2, img_w, img_h)

        if cls_id == settings.CLASS_TEXT_LINE:
            is_valid, reason = _is_text_detection_valid(det, img_w, img_h)
            if not is_valid:
                profile["counts"]["detections_filtered"] += 1
                profile["filters"]["text_dropped_reasons"][reason] = (
                    profile["filters"]["text_dropped_reasons"].get(reason, 0) + 1
                )
                continue

            if profile["ocr"]["used_engine"] == "google":
                matched_tokens = get_text_in_zone(bbox, google_tokens)
                token = _concat_google_tokens(matched_tokens)
                if token != "[VIDE]":
                    profile["ocr"]["google_matched_tokens"] += len(matched_tokens)
                entries.append({
                    "type": "text",
                    "token": token,
                    "bbox": bbox,
                    "bbox_norm": norm_bb,
                    "yolo_conf": float(det["conf"]),
                })
                continue

            if trocr_calls_planned >= settings.MAX_TROCR_ZONES:
                profile["filters"]["trocr_skipped_by_limit"] += 1
                continue

            crop_cv = get_rotated_crop(img_cv, det["pts"])
            if crop_cv.size == 0 or crop_cv.shape[0] < 5 or crop_cv.shape[1] < 5:
                profile["counts"]["detections_filtered"] += 1
                profile["filters"]["text_dropped_reasons"]["invalid_crop"] = (
                    profile["filters"]["text_dropped_reasons"].get("invalid_crop", 0) + 1
                )
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop_cv, cv2.COLOR_BGR2RGB))
            crop_pil = _prepare_trocr_crop(crop_pil)
            entries.append({
                "type": "text",
                "bbox": bbox,
                "bbox_norm": norm_bb,
                "yolo_conf": float(det["conf"]),
            })
            text_crops.append(crop_pil)
            trocr_calls_planned += 1

        elif cls_id == settings.CLASS_CHECKBOX_CHECKED:
            if float(det["conf"]) < settings.CHECKBOX_MIN_CONF:
                profile["counts"]["detections_filtered"] += 1
                continue
            entries.append({
                "type": "token",
                "token": "OUI",
                "bbox": bbox,
                "bbox_norm": norm_bb,
                "yolo_conf": float(det["conf"]),
            })

        elif cls_id == settings.CLASS_CHECKBOX_EMPTY:
            if not settings.MAP_CHECKBOX_EMPTY_TO_NON:
                continue
            if float(det["conf"]) < settings.CHECKBOX_MIN_CONF:
                profile["counts"]["detections_filtered"] += 1
                continue
            entries.append({
                "type": "token",
                "token": "NON",
                "bbox": bbox,
                "bbox_norm": norm_bb,
                "yolo_conf": float(det["conf"]),
            })

        else:
            continue

    if not entries:
        logger.warning("Pipeline : aucune entree exploitable")
        return [], profile if collect_profile else {}

    # 3. OCR TrOCR batch sur les zones texte
    text_tokens = []
    if text_crops and profile["ocr"]["used_engine"] != "google":
        t_trocr = perf_counter()
        text_tokens, trocr_meta = run_trocr_batch(text_crops, registry)
        profile["timings_s"]["trocr"] = round(perf_counter() - t_trocr, 6)
        profile["counts"]["trocr_calls"] = int(trocr_meta["processed"])
        profile["filters"]["trocr_skipped_by_timeout"] = int(trocr_meta["skipped_timeout"])
        profile["filters"]["trocr_errors"] = int(trocr_meta["errors"])
        profile["timings_s"]["ocr"] = profile["timings_s"]["trocr"]
        profile["ocr"]["used_engine"] = "trocr"
        if float(settings.TROCR_REFERENCE_OCR_S) > 0:
            profile["ocr"]["gain_vs_trocr_s"] = round(float(settings.TROCR_REFERENCE_OCR_S) - profile["timings_s"]["trocr"], 6)

    raw_tokens = []
    raw_bboxes = []
    raw_norm_bb = []
    raw_yolo_conf = []
    text_idx = 0

    for entry in entries:
        if entry["type"] == "token":
            token = entry["token"]
        else:
            # En mode Google, le token est deja calcule au moment du matching OBB.
            if "token" in entry:
                token = entry["token"]
            else:
                token = text_tokens[text_idx] if text_idx < len(text_tokens) else OCR_TIMEOUT_TOKEN
                text_idx += 1

                if token in (OCR_TIMEOUT_TOKEN, "[ERREUR]"):
                    profile["counts"]["detections_filtered"] += 1
                    profile["filters"]["text_dropped_reasons"]["trocr_unusable"] = (
                        profile["filters"]["text_dropped_reasons"].get("trocr_unusable", 0) + 1
                    )
                    continue

        raw_tokens.append(token)
        raw_bboxes.append(entry["bbox"])
        raw_norm_bb.append(entry["bbox_norm"])
        raw_yolo_conf.append(float(entry.get("yolo_conf", 0.0)))

    if not raw_tokens:
        logger.warning("Pipeline : aucun token extrait")
        return [], profile if collect_profile else {}

    logger.info(f"Pipeline : {len(raw_tokens)} tokens — lancement LayoutLMv3")

    # 4. LayoutLMv3
    t_layout = perf_counter()
    ner_labels, ner_scores = run_layoutlm(work_img, raw_tokens, raw_norm_bb, registry)
    profile["timings_s"]["layoutlm"] = round(perf_counter() - t_layout, 6)
    profile["counts"]["tokens"] = len(raw_tokens)

    # Assemblage
    t_assemble = perf_counter()
    results = []
    for token, bbox, norm_bb, ner, y_conf, n_conf in zip(
        raw_tokens,
        raw_bboxes,
        raw_norm_bb,
        ner_labels,
        raw_yolo_conf,
        ner_scores,
    ):
        results.append({
            "token":     token,
            "bbox":      bbox,
            "bbox_norm": norm_bb,
            "ner_label": ner,
            "yolo_conf": round(float(y_conf), 4),
            "ner_conf":  round(float(n_conf), 4),
        })

    profile["timings_s"]["trocr"] = round(profile["timings_s"]["trocr"], 6)
    profile["timings_s"]["assemble"] = round(perf_counter() - t_assemble, 6)

    pipeline_elapsed = perf_counter() - t_pipeline
    logger.info(f"Pipeline complet : {pipeline_elapsed:.3f}s")

    return results, profile if collect_profile else {}
