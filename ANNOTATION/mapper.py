"""Mapper metier des predictions NER vers les champs formulaire."""

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import unicodedata

from core.logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]



DEFAULT_POSITION_RULES: dict[str, list[tuple[int, int, str]]] = {

    "IDENTITE": [
        (0,    60,   "type_deces"),
        (60,   120,  "enqueteur"),
        (120,  170,  "repondant"),
        (170,  220,  "ou_menage_identite"),
        (220,  280,  "mere"),
        (280,  340,  "sexe"),
        (700,  780,  "nom_medecin"),
    ],

    "DATE": [
        (60,   120,  "date_visite"),
        (280,  340,  "date_naissance"),
        (340,  400,  "date_deces"),
        (500,  560,  "date_consultation_etablissement"),
        (620,  680,  "date_etablissement_hospitalise"),
    ],

    "LIEU": [
        (120,  170,  "village"),
        (170,  220,  "concession"),
        (400,  460,  "lieu_deces"),
        (460,  520,  "lieu_deces_precise"),
    ],

    "ETABLISSEMENT": [
        (500,  560,  "quel_etablissement_consultation"),
        (560,  620,  "lieu_consultation_etablissement"),
        (620,  680,  "quel_etablissement_hospitalise"),
        (680,  740,  "lieu_etablissement_hospitalise"),
    ],

    "SYMPTOME": [
        (0,    100,  "_premier_symptome"),
        (100,  200,  "maladie_avant_premier_symptome"),
        (200,  300,  "fievre_ou_corps_chaud"),
        (300,  400,  "toux"),
        (400,  500,  "diarrhee_dysenterie"),
        (500,  600,  "autres_symptomes"),
        (600,  700,  "autres_personnes_symptomes"),
    ],

    "DUREE": [
        (800,  900,  "duree_jusqua_deces"),
        (100,  200,  "duree_maladie_conduit_deces"),
        (200,  300,  "duree_fievre"),
        (300,  400,  "duree_toux"),
        (400,  500,  "duree_vomissements"),
    ],

    "TRAITEMENT": [
        (0,    500,  "traitement_medicament_soins"),
        (500,  1000, "traitement_lieu_qui"),
    ],

    "DIAGNOSTIC": [
        (740,  800,  "diagnostic_registre"),
        (800,  860,  "cause_initiale_cim10"),
        (860,  920,  "cause_declaree"),
        (920,  980,  "nom_local_maladie"),
        (980,  1000, "type_deces"),
    ],

    "REPONSE": [
        (460,  530,  "consulte_guerisseur"),
        (530,  600,  "consulte_etablissement"),
        (600,  670,  "hospitalise"),
        (670,  740,  "sagit_il_accident"),
        (740,  810,  "si_accident_utilise_cim9"),
    ],

    "AUTRE": [
        (0,    1000, "histoire_symptomes_traitements"),
    ],
}


PAGE_RULE_OVERRIDES: dict[int, dict[str, list[tuple[int, int, str]]]] = {}


DATE_RE = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$")
HAS_DIGIT_RE = re.compile(r"\d")
DATE_ANY_SEP_RE = re.compile(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})")
LABEL_CLASS_LINE_RE = re.compile(r'^\s*"([^"]+)"\s*:\s*"([A-Z_]+)"\s*,?\s*$')
GENERIC_CLASS_LABELS = {
    "identite", "date", "lieu", "etablissement", "symptome",
    "duree", "traitement", "diagnostic", "reponse", "autre",
}


def _infer_class_from_label(label: str) -> str | None:
    l = label.lower()
    if "date" in l:
        return "DATE"
    if l.startswith("duree_") or l.startswith("debut_") or l.startswith("fin_") or l.startswith("age_"):
        return "DUREE"
    if any(k in l for k in ["village", "concession", "lieu_deces", "lieu_"]):
        return "LIEU"
    if any(k in l for k in ["etablissement", "centre_sante", "formation_sanitaire"]):
        return "ETABLISSEMENT"
    if any(k in l for k in ["hospitalise", "consulte", "accident", "oui_non", "reponse"]):
        return "REPONSE"
    if any(k in l for k in ["diagnostic", "cause_", "cim", "nom_local_maladie", "type_deces"]):
        return "DIAGNOSTIC"
    if any(k in l for k in ["traitement", "medicament", "injection", "soins"]):
        return "TRAITEMENT"
    if any(k in l for k in ["histoire_", "remarque", "description_"]):
        return "AUTRE"
    if any(k in l for k in ["nom", "identite", "mere", "enqueteur", "repondant", "sexe", "numero_"]):
        return "IDENTITE"
    return None


def _load_label_to_class_from_markdown() -> dict[str, str]:
    mapping: dict[str, str] = {}
    md_path = ROOT_DIR / "LABEL_MAPPING.md"
    if not md_path.exists():
        logger.warning("LABEL_MAPPING.md introuvable: mapping auto des coordonnees desactive")
        return mapping

    for line in md_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LABEL_CLASS_LINE_RE.match(line)
        if not m:
            continue
        label = m.group(1).strip()
        classe = m.group(2).strip()
        mapping[label] = classe
    return mapping


def _build_page_overrides_from_coordinates() -> dict[int, dict[str, list[tuple[int, int, str]]]]:
    """
    Construit des regles par page (1..4) depuis Coordonne_des_labels/*.json.
    Les coordonnees Label Studio sont en pourcentage [0-100].
    On convertit en [0-1000] et on ajoute une marge pour robustesse.
    """
    coord_dir = ROOT_DIR / "Coordonne_des_labels"
    if not coord_dir.exists():
        logger.warning("Coordonne_des_labels introuvable: fallback sur regles par defaut")
        return {}

    label_to_class = _load_label_to_class_from_markdown()
    overrides: dict[int, dict[str, list[tuple[int, int, str]]]] = {}

    for page_number in [1, 2, 3, 4]:
        path = coord_dir / f"coordonnees_page{page_number}.json"
        if not path.exists():
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as e:
            logger.warning(f"Lecture impossible {path.name}: {e}")
            continue

        annotations = []
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            annotations = payload[0].get("annotations", [])

        by_class_label: dict[str, dict[str, list[tuple[float, float]]]] = {}

        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            labels = ann.get("labels", [])
            if not labels:
                continue

            label = str(labels[0]).strip()
            if not label:
                continue

            classe = label_to_class.get(label) or _infer_class_from_label(label)
            if not classe:
                continue

            y = float(ann.get("y", 0.0))
            h = float(ann.get("height", 0.0))
            center_norm = (y + (h / 2.0)) * 10.0
            height_norm = max(5.0, h * 10.0)

            by_class_label.setdefault(classe, {}).setdefault(label, []).append((center_norm, height_norm))

        page_rules: dict[str, list[tuple[int, int, str]]] = {}

        for classe, labels_data in by_class_label.items():
            rules = []
            for label, vals in labels_data.items():
                centers = [v[0] for v in vals]
                heights = [v[1] for v in vals]
                c = sum(centers) / len(centers)
                h = sum(heights) / len(heights)

                # Marge robuste: au moins 25 points norm, et 2.5x hauteur moyenne
                margin = max(25, int(round(h * 2.5)))
                y_min = max(0, int(round(c - margin)))
                y_max = min(1000, int(round(c + margin)))
                rules.append((y_min, y_max, label))

            rules.sort(key=lambda t: (t[0] + t[1]) / 2.0)
            if rules:
                page_rules[classe] = rules

        if page_rules:
            overrides[page_number] = page_rules

    return overrides


PAGE_RULE_OVERRIDES = _build_page_overrides_from_coordinates()


def _get_rules_for_page(page_number: int) -> dict[str, list[tuple[int, int, str]]]:
    rules = {k: list(v) for k, v in DEFAULT_POSITION_RULES.items()}
    overrides = PAGE_RULE_OVERRIDES.get(page_number, {})
    for classe, new_rules in overrides.items():
        rules[classe] = new_rules + rules.get(classe, [])
    return rules


def _get_real_label(classe: str, y_norm: int, page_number: int) -> str:
    """
    Retourne le vrai label selon la classe et la position Y normalisee.
    Fallback sur la classe elle-meme si aucune regle ne correspond.
    """
    rules = _get_rules_for_page(page_number).get(classe, [])
    for y_min, y_max, label in rules:
        if y_min <= y_norm <= y_max:
            return label
    return classe.lower()


def _is_valid_date(text: str) -> bool:
    m = DATE_RE.match(text)
    if not m:
        return False
    day = int(m.group(1))
    month = int(m.group(2))
    year = int(m.group(3))
    if year < 100:
        year += 2000
    return 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100


def _sanitize_text(text: str) -> str:
    if not text:
        return ""

    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\ufffd", "")
    t = "".join(ch for ch in t if ch.isprintable())

    # Nettoyage artefacts OCR frequents
    t = t.replace("|", "/")
    t = re.sub(r"([/\\:;,_\-.])\1+", r"\1", t)
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" .,:;|_-/")
    return t.strip()


def _normalize_date(text: str) -> str:
    m = DATE_ANY_SEP_RE.search(text)
    if not m:
        return text

    day = int(m.group(1))
    month = int(m.group(2))
    year = int(m.group(3))
    if year < 100:
        year += 2000

    normalized = f"{day:02d}/{month:02d}/{year:04d}"
    return normalized if _is_valid_date(normalized) else text


def _normalize_yes_no(text: str) -> str:
    if not text:
        return text
    up = unicodedata.normalize("NFKC", text).upper()
    compact = re.sub(r"[^A-Z0-9]", "", up)

    if compact in {"OUI", "0UI", "OU1", "QUI", "QUL"} or compact.startswith("OU"):
        return "OUI"
    if compact in {"NON", "N0N", "NQN", "NOM"} or compact.startswith("NO"):
        return "NON"
    return text


def _text_quality_score(text: str) -> float:
    if not text:
        return 0.0
    if text in {"[ERREUR]", "[TIMEOUT]", "[VIDE]"}:
        return 0.0

    printable = sum(1 for ch in text if ch.isprintable())
    ratio_printable = printable / max(1, len(text))
    weird = text.count("?") + text.count("�")
    weird_ratio = weird / max(1, len(text))
    has_digit = 1.0 if HAS_DIGIT_RE.search(text) else 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    alpha_ratio = alpha / max(1, len(text))

    score = (0.55 * ratio_printable) + (0.2 * (1.0 - weird_ratio)) + (0.15 * alpha_ratio) + (0.1 * has_digit)
    return max(0.0, min(1.0, score))


def _format_score(label: str, text: str) -> tuple[float, str, bool]:
    low_label = label.lower()
    txt = text.strip()

    if "date" in low_label:
        digits = sum(1 for ch in txt if ch.isdigit())
        if digits > 8:
            return 0.05, "date", False
        valid = _is_valid_date(txt)
        return (1.0 if valid else 0.2), "date", valid

    if "identite" in low_label:
        digits = sum(1 for ch in txt if ch.isdigit())
        valid = digits >= 4
        return (0.9 if valid else 0.35), "identifiant", valid

    if any(k in low_label for k in ["hospitalise", "consulte", "accident", "reponse"]):
        valid = txt.upper() in {"OUI", "NON"}
        return (1.0 if valid else 0.4), "oui_non", valid

    return 0.65, "none", True


@dataclass
class Entity:
    tokens:    list[str]   = field(default_factory=list)
    bboxes:    list[list]  = field(default_factory=list)
    yolo_confs: list[float] = field(default_factory=list)
    ner_confs:  list[float] = field(default_factory=list)
    classe:    str         = ""
    real_label: str        = ""
    confidence: float      = 0.0
    format_type: str       = "none"
    format_valid: bool     = True

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def y_center_norm(self) -> int:
        """Y centre moyen de toutes les bboxes (coordonnees norm [0-1000])."""
        if not self.bboxes:
            return 0
        return int(sum((bb[1] + bb[3]) / 2 for bb in self.bboxes) / len(self.bboxes))

    @property
    def yolo_conf_mean(self) -> float:
        return float(sum(self.yolo_confs) / len(self.yolo_confs)) if self.yolo_confs else 0.0

    @property
    def ner_conf_mean(self) -> float:
        return float(sum(self.ner_confs) / len(self.ner_confs)) if self.ner_confs else 0.0


def map_predictions(predictions: list[dict], page_number: int = 1) -> dict:
    """
    Convertit la liste de predictions NER en JSON final structure.

    Entree :
        [
          {"token": "Tapo Nickhon", "bbox": [...], "bbox_norm": [...], "ner_label": "B-IDENTITE"},
          {"token": "01/03/2026",   "bbox": [...], "bbox_norm": [...], "ner_label": "B-DATE"},
          ...
        ]

    Sortie :
        {
          "enqueteur":   "Tapo Nickhon",
          "date_visite": "01/03/2026",
          ...
          "_meta": {
              "entities_count": 12,
              "classes_found": ["IDENTITE", "DATE", ...]
          }
        }
    """
    if not predictions:
        return {}

    # ── 1. Grouper les tokens en entites (BIO) ─────────────────────────────
    entities: list[Entity] = []
    current: Entity | None = None

    for pred in predictions:
        ner  = pred.get("ner_label", "O")
        if ner == "O":
            if current:
                entities.append(current)
                current = None
            continue

        tag   = ner[0]      # B ou I
        classe = ner[2:]    # IDENTITE, DATE, etc.

        if tag == "B":
            if current:
                entities.append(current)
            current = Entity(
                tokens=[pred["token"]],
                bboxes=[pred["bbox_norm"]],
                yolo_confs=[float(pred.get("yolo_conf", 0.0))],
                ner_confs=[float(pred.get("ner_conf", 0.0))],
                classe=classe,
            )
        elif tag == "I" and current and current.classe == classe:
            current.tokens.append(pred["token"])
            current.bboxes.append(pred["bbox_norm"])
            current.yolo_confs.append(float(pred.get("yolo_conf", 0.0)))
            current.ner_confs.append(float(pred.get("ner_conf", 0.0)))

    if current:
        entities.append(current)

    # ── 2. Assigner les vrais labels via position Y ─────────────────────────
    output: dict[str, str | list | dict] = {}
    output_conf: dict[str, dict] = {}
    classes_found = set()
    duplicate_dropped = 0
    normalization_changes = 0
    uncertain_fields = 0

    for entity in entities:
        entity.real_label = _get_real_label(entity.classe, entity.y_center_norm, page_number)
        classes_found.add(entity.classe)

        label = entity.real_label
        raw_text = entity.text
        text = _sanitize_text(raw_text)

        if "date" in label.lower():
            text = _normalize_date(text)
        if any(k in label.lower() for k in ["hospitalise", "consulte", "accident", "reponse"]):
            text = _normalize_yes_no(text)

        if text != raw_text:
            normalization_changes += 1

        text_q = _text_quality_score(text)
        fmt_s, fmt_type, fmt_valid = _format_score(label, text)
        entity.format_type = fmt_type
        entity.format_valid = fmt_valid

        # Score agrege champ par champ
        entity.confidence = max(
            0.0,
            min(
                1.0,
                (0.30 * entity.yolo_conf_mean)
                + (0.35 * entity.ner_conf_mean)
                + (0.20 * text_q)
                + (0.15 * fmt_s),
            ),
        )

        # Si plusieurs valeurs pour un label, garder la plus fiable
        had_existing = label in output
        if (not had_existing) or entity.confidence >= output_conf.get(label, {}).get("score", 0.0):
            is_uncertain = (entity.confidence < 0.55) or (not fmt_valid)
            output[label] = text
            output_conf[label] = {
                "score": round(entity.confidence, 4),
                "uncertain": is_uncertain,
                "classe": entity.classe,
                "y_center_norm": entity.y_center_norm,
                "format_type": fmt_type,
                "format_valid": fmt_valid,
                "yolo_conf": round(entity.yolo_conf_mean, 4),
                "ner_conf": round(entity.ner_conf_mean, 4),
            }
            if had_existing:
                duplicate_dropped += 1
        else:
            duplicate_dropped += 1

    # Supprime les labels generiques (ex: "reponse", "date")
    # lorsqu'un label plus specifique de la meme classe est deja present.
    for generic in list(GENERIC_CLASS_LABELS):
        if generic not in output_conf:
            continue
        generic_class = output_conf[generic].get("classe")
        if not generic_class:
            continue

        has_specific = any(
            key != generic and meta.get("classe") == generic_class
            for key, meta in output_conf.items()
        )
        if has_specific:
            output.pop(generic, None)
            output_conf.pop(generic, None)

    uncertain_fields = sum(1 for meta in output_conf.values() if bool(meta.get("uncertain")))

    # ── 3. Metadata ────────────────────────────────────────────────────────
    output["_confidence"] = output_conf
    output["_meta"] = {
        "entities_count":  len(entities),
        "classes_found":   sorted(classes_found),
        "tokens_processed": len(predictions),
        "page_number": page_number,
        "duplicate_dropped": duplicate_dropped,
        "normalization_changes": normalization_changes,
        "uncertain_fields": uncertain_fields,
    }

    logger.info(f"Mapper : {len(entities)} entites → {max(0, len(output)-2)} champs extraits")
    return output
