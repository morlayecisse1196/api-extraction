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
PAGE_2D_RULES: dict[int, dict[str, list[tuple[int, int, int, int, str]]]] = {}


DATE_RE = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\s*$")
HAS_DIGIT_RE = re.compile(r"\d")
DATE_ANY_SEP_RE = re.compile(r"(\d{1,2})\D+(\d{1,2})\D+(\d{2,4})")
LABEL_CLASS_LINE_RE = re.compile(r'^\s*"([^"]+)"\s*:\s*"([A-Z_]+)"\s*,?\s*$')
GENERIC_CLASS_LABELS = {
    "identite", "date", "lieu", "etablissement", "symptome",
    "duree", "traitement", "diagnostic", "reponse", "autre",
}

MIN_CONFIDENCE_BY_FORMAT = {
    "date": 0.82,
    "oui_non": 0.70,
    "identifiant": 0.72,
    "none": 0.58,
}

PAGE_LABEL_SPECS: dict[int, list[tuple[str, str]]] = {
    1: [
        ("type_deces", "REPONSE"),
        ("enqueteur", "IDENTITE"),
        ("date_visite", "DATE"),
        ("village", "LIEU"),
        ("concession", "LIEU"),
        ("ou_menage_identite", "IDENTITE"),
        ("mere", "IDENTITE"),
        ("sexe", "IDENTITE"),
        ("date_naissance", "DATE"),
        ("date_deces", "DATE"),
        ("repondant", "IDENTITE"),
        ("age_declare", "DUREE"),
        ("lieu_deces", "LIEU"),
        ("lieu_deces_precise", "LIEU"),
        ("cause_declaree", "DIAGNOSTIC"),
        ("nom_local_maladie", "DIAGNOSTIC"),
        ("consulte_guerisseur", "REPONSE"),
        ("consulte_etablissement", "REPONSE"),
        ("quel_etablissement_consultation", "ETABLISSEMENT"),
        ("lieu_consultation_etablissement", "ETABLISSEMENT"),
        ("date_consultation_etablissement", "DATE"),
        ("hospitalise", "REPONSE"),
        ("quel_etablissement_hospitalise", "ETABLISSEMENT"),
        ("lieu_etablissement_hospitalise", "ETABLISSEMENT"),
        ("date_etablissement_hospitalise", "DATE"),
        ("diagnostic_registre", "DIAGNOSTIC"),
        ("cause_initiale_cim10", "DIAGNOSTIC"),
        ("duree_jusqua_deces", "DUREE"),
        ("si_accident_utilise_cim9", "REPONSE"),
    ],
    2: [
        ("sagit_il_accident", "REPONSE"),
        ("histoire_symtomes_traitements", "AUTRE"),
        ("_premier_symptome", "SYMPTOME"),
        ("maladie_avant_premier_symptome", "SYMPTOME"),
        ("duree_maladie_conduit_deces", "DUREE"),
        ("traitement_medicament_soins", "TRAITEMENT"),
        ("traitement_lieu_qui", "ETABLISSEMENT"),
        ("traitement_date", "DATE"),
        ("autres_personnes_symptomes", "SYMPTOME"),
        ("village_autres_personnes", "LIEU"),
    ],
    3: [
        ("fievre_ou_corps_chaud", "SYMPTOME"),
        ("duree_fievre", "DUREE"),
        ("debut_fievre", "DUREE"),
        ("fin_fievre", "DUREE"),
        ("fievre_forte", "SYMPTOME"),
        ("fievre_moyenne", "SYMPTOME"),
        ("fievre_intermittente", "SYMPTOME"),
        ("fievre_continue", "SYMPTOME"),
        ("sueurs", "SYMPTOME"),
        ("frissons", "SYMPTOME"),
        ("act_antipaludeens_donne", "TRAITEMENT"),
        ("act_antipaludeens_nombre_fois", "TRAITEMENT"),
        ("act_antipaludeens_nombre_comprimes", "TRAITEMENT"),
        ("autre_traitement_oui_non", "REPONSE"),
        ("autre_traitement_nom", "TRAITEMENT"),
        ("injection_fievre_oui_non", "REPONSE"),
        ("injection_fievre_lieu", "ETABLISSEMENT"),
        ("injection_fievre_date", "DATE"),
        ("diarrhee_dysenterie", "SYMPTOME"),
        ("duree_selles", "DUREE"),
        ("debut_selles", "DUREE"),
        ("fin_selles", "DUREE"),
        ("nb_selles_jour", "SYMPTOME"),
        ("selles_comme_eau_incolore", "SYMPTOME"),
        ("selles_comme_craches", "SYMPTOME"),
        ("selles_avec_sang", "SYMPTOME"),
        ("selles_autre_couleur", "SYMPTOME"),
        ("selles_autre_couleur_preciser", "SYMPTOME"),
        ("signes_deshydratation", "SYMPTOME"),
        ("bouche_langue_seche", "SYMPTOME"),
        ("yeux_enfonces", "SYMPTOME"),
        ("fontanelle_deprimee", "SYMPTOME"),
    ],
    4: [
        ("vomissements", "SYMPTOME"),
        ("duree_vomissements", "DUREE"),
        ("moment_vomissements", "SYMPTOME"),
        ("couleur_vomissements", "SYMPTOME"),
        ("vomissements_en_jet", "SYMPTOME"),
        ("crises_convulsives", "SYMPTOME"),
        ("nombre_crises", "SYMPTOME"),
        ("duree_crise", "DUREE"),
        ("moment_crises", "SYMPTOME"),
        ("spasmes_mouvement_brusque_incontrole", "SYMPTOME"),
        ("cris_pleurs", "SYMPTOME"),
        ("urine", "SYMPTOME"),
        ("morsure_langue", "SYMPTOME"),
        ("hypersalivation", "SYMPTOME"),
        ("respiration_bruyante_crise", "SYMPTOME"),
        ("fontanelle_gonflee", "SYMPTOME"),
        ("cou_tordu", "SYMPTOME"),
        ("corps_raidi", "SYMPTOME"),
        ("jambes_tendues", "SYMPTOME"),
        ("jambes_pliees", "SYMPTOME"),
        ("bras_tendus", "SYMPTOME"),
        ("bras_plies", "SYMPTOME"),
        ("poings_fermes", "SYMPTOME"),
        ("bouche_fermee_ou_crispee", "SYMPTOME"),
        ("perte_connaissance", "SYMPTOME"),
        ("duree_perte_moins_1h", "DUREE"),
        ("duree_perte_plus_1h", "DUREE"),
        ("epilepsie", "DIAGNOSTIC"),
        ("duree_crises", "DUREE"),
        ("soigne_oui_non", "REPONSE"),
        ("lieu_soin_epilepsie", "ETABLISSEMENT"),
        ("signes_neurologiques_contexte_crises", "REPONSE"),
        ("perte_connaissance_coma", "SYMPTOME"),
        ("moment_perte_connaissance", "SYMPTOME"),
        ("duree_perte_connaissance_coma", "DUREE"),
        ("paralysie", "SYMPTOME"),
        ("parties_paralysie", "SYMPTOME"),
        ("duree_paralysie", "DUREE"),
    ],
}

HISTORY_LABEL_ALIASES = ["histoire_symtomes_traitements", "histoire_symptomes_traitements"]
CIM_CODE_RE = re.compile(r"^\s*[A-Z]?[0-9]{1,2}(?:[\.,][0-9]{1,3})?\s*$", re.IGNORECASE)


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


def _build_page_2d_rules_from_coordinates() -> dict[int, dict[str, list[tuple[int, int, int, int, str]]]]:
    """
    Construit des zones 2D (x_min, y_min, x_max, y_max) par page et par classe
    depuis Coordonne_des_labels/*.json.

    Les valeurs Label Studio sont en pourcentage [0-100], converties en [0-1000].
    On ajoute une marge autour de chaque annotation pour tolérer les écarts.
    """
    coord_dir = ROOT_DIR / "Coordonne_des_labels"
    if not coord_dir.exists():
        return {}

    label_to_class = _load_label_to_class_from_markdown()
    page_rules: dict[int, dict[str, list[tuple[int, int, int, int, str]]]] = {}

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

        rules_by_class: dict[str, list[tuple[int, int, int, int, str]]] = {}

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

            x = float(ann.get("x", 0.0))
            y = float(ann.get("y", 0.0))
            width = float(ann.get("width", 0.0))
            height = float(ann.get("height", 0.0))

            x_min = max(0, int(round(x * 10)))
            y_min = max(0, int(round(y * 10)))
            x_max = min(1000, int(round((x + width) * 10)))
            y_max = min(1000, int(round((y + height) * 10)))

            # Marge adaptative: volontairement resserree pour limiter les fusions
            # sur les champs larges, avec une tolerance minimale.
            margin_x = max(12, min(24, int(round(width * 10 * 0.12))))
            margin_y = max(12, min(18, int(round(height * 10 * 0.85))))

            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(1000, x_max + margin_x)
            y_max = min(1000, y_max + margin_y)

            rules_by_class.setdefault(classe, []).append((x_min, y_min, x_max, y_max, label))

        if rules_by_class:
            page_rules[page_number] = rules_by_class

    return page_rules


PAGE_RULE_OVERRIDES = _build_page_overrides_from_coordinates()
PAGE_2D_RULES = _build_page_2d_rules_from_coordinates()


def _merge_default_with_page1(
    base_rules: dict[str, list[tuple[int, int, str]]],
    page1_rules: dict[str, list[tuple[int, int, str]]] | None,
) -> dict[str, list[tuple[int, int, str]]]:
    """
    Utilise les coordonnees reelles de la page 1 comme base DEFAULT,
    puis conserve les regles historiques en fallback.
    """
    if not page1_rules:
        return base_rules

    merged = {k: list(v) for k, v in base_rules.items()}
    for classe, rules in page1_rules.items():
        # Priorite aux vraies coordonnees label studio de la page 1
        merged[classe] = list(rules) + merged.get(classe, [])
    return merged


# IMPORTANT:
# DEFAULT_POSITION_RULES devient "default calibre" a partir des vraies
# coordonnees de la page 1 (si disponibles), avec fallback historique.
DEFAULT_POSITION_RULES = _merge_default_with_page1(
    DEFAULT_POSITION_RULES,
    PAGE_RULE_OVERRIDES.get(1),
)


def _get_rules_for_page(page_number: int) -> dict[str, list[tuple[int, int, str]]]:
    rules = {k: list(v) for k, v in DEFAULT_POSITION_RULES.items()}
    overrides = PAGE_RULE_OVERRIDES.get(page_number, {})
    for classe, new_rules in overrides.items():
        rules[classe] = new_rules + rules.get(classe, [])
    return rules


def _get_2d_rules_for_page(page_number: int) -> dict[str, list[tuple[int, int, int, int, str]]]:
    return PAGE_2D_RULES.get(page_number, {})


def _get_real_label_2d(classe: str, x_norm: int, y_norm: int, page_number: int) -> str | None:
    rules = _get_2d_rules_for_page(page_number).get(classe, [])
    for x_min, y_min, x_max, y_max, label in rules:
        if x_min <= x_norm <= x_max and y_min <= y_norm <= y_max:
            return label
    return None


def _get_real_label_with_source(classe: str, x_norm: int, y_norm: int, page_number: int) -> tuple[str, str]:
    """
    Retourne (label, source_mapping) ou source_mapping ∈ {"2d", "1d", "class_fallback"}.
    """
    label_2d = _get_real_label_2d(classe, x_norm, y_norm, page_number)
    if label_2d:
        return label_2d, "2d"

    rules = _get_rules_for_page(page_number).get(classe, [])
    for y_min, y_max, label in rules:
        if y_min <= y_norm <= y_max:
            return label, "1d"

    return classe.lower(), "class_fallback"


def _get_real_label(classe: str, x_norm: int, y_norm: int, page_number: int) -> str:
    """
    Retourne le vrai label selon la classe et la position Y normalisee.
    Fallback sur la classe elle-meme si aucune regle ne correspond.
    """
    label, _ = _get_real_label_with_source(classe, x_norm, y_norm, page_number)
    return label


def _min_confidence_for_format(fmt_type: str) -> float:
    return float(MIN_CONFIDENCE_BY_FORMAT.get(fmt_type, MIN_CONFIDENCE_BY_FORMAT["none"]))


def _looks_like_ocr_noise(label: str, text: str) -> bool:
    if not text:
        return True

    cleaned = text.strip()
    if len(cleaned) <= 1:
        return True

    digits = sum(1 for ch in cleaned if ch.isdigit())
    letters = sum(1 for ch in cleaned if ch.isalpha())
    separators = sum(1 for ch in cleaned if ch in "/\\|:;,_-.")
    total = max(1, len(cleaned))
    sep_ratio = separators / total
    digit_ratio = digits / total

    low_label = label.lower()

    if "date" in low_label:
        return not _is_valid_date(cleaned)

    if any(k in low_label for k in ["hospitalise", "consulte", "accident", "reponse"]):
        return cleaned.upper() not in {"OUI", "NON"}

    # Pour les labels texte, rejette les chaines trop segmentees et numeriques.
    if letters == 0:
        return True
    if sep_ratio > 0.22 and digit_ratio > 0.20:
        return True

    return False


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
    groups = [int(part) for part in re.findall(r"\d{1,4}", text)]
    if len(groups) < 3:
        m = DATE_ANY_SEP_RE.search(text)
        if not m:
            return text
        groups = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    candidates: list[tuple[int, int, int]] = []
    if len(groups) >= 3:
        a, b, c = groups[:3]
        candidates.extend([
            (a, b, c),
            (b, c, a),
            (a, c, b),
        ])

        # OCR courant: annee en premier, puis jour/mois.
        if a >= 1000:
            candidates.append((b, c, a))
        if c >= 1000:
            candidates.append((a, b, c))
        if b >= 1000:
            candidates.append((a, c, b))

    seen: set[tuple[int, int, int]] = set()
    for day, month, year in candidates:
        key = (day, month, year)
        if key in seen:
            continue
        seen.add(key)
        if year < 100:
            year += 2000
        normalized = f"{day:02d}/{month:02d}/{year:04d}"
        if _is_valid_date(normalized):
            return normalized

    return text


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


def _normalize_sexe_value(text: str) -> str:
    if not text:
        return "indefini"
    up = unicodedata.normalize("NFKC", text).upper()
    compact = re.sub(r"[^A-Z]", "", up)

    if compact in {"F", "FEMME", "FILLE", "FEMININ"}:
        return "F"
    if compact in {"M", "H", "HOMME", "MASCULIN", "GARCON"}:
        return "M"

    # Robustesse OCR: repere presence dominante de F ou M.
    has_f = "F" in compact
    has_m = "M" in compact or "H" in compact
    if has_f and not has_m:
        return "F"
    if has_m and not has_f:
        return "M"
    return "indefini"


def _normalize_type_deces_value(text: str) -> str:
    if not text:
        return "indefini"
    up = unicodedata.normalize("NFKC", text).upper()
    compact = re.sub(r"\s+", " ", up).strip()

    options = [
        ("NOUVEAU-NE", ["NOUVEAU", "NE", "NOURRISSON"]),
        ("ENFANT", ["ENFANT"]),
        ("ADULTE FEMME", ["ADULTE", "FEMME"]),
        ("ADULTE HOMME", ["ADULTE", "HOMME"]),
        ("MATERNEL", ["MATERNEL"]),
    ]

    for normalized, keys in options:
        if all(k in compact for k in keys):
            return normalized

    if "ADULTE" in compact and "F" in compact and "H" not in compact:
        return "ADULTE FEMME"
    if "ADULTE" in compact and ("H" in compact or "M" in compact):
        return "ADULTE HOMME"
    return "indefini"


def _normalize_date_or_indefini(text: str) -> str:
    cleaned = _sanitize_text(text)
    normalized = _normalize_date(cleaned)
    return normalized if _is_valid_date(normalized) else "indefini"


def _default_value_for_class(classe: str) -> str:
    return "NON" if classe == "REPONSE" else "indefini"


def _extract_candidate_text(label: str, extracted_values: dict[str, str]) -> str:
    if label in extracted_values:
        return extracted_values[label]
    if label in HISTORY_LABEL_ALIASES:
        for alt in HISTORY_LABEL_ALIASES:
            if alt in extracted_values:
                return extracted_values[alt]
    return ""


def _is_placeholder_value(value: str) -> bool:
    if not isinstance(value, str):
        return True
    compact = value.strip()
    return compact in {"", "[VIDE]", "[ERREUR]", "[TIMEOUT]", "indefini"}


def _accept_token_for_expected_class(expected_class: str, token: str) -> bool:
    txt = (token or "").strip()
    if not txt or txt in {"[VIDE]", "[ERREUR]", "[TIMEOUT]"}:
        return False

    up = txt.upper()
    if expected_class == "REPONSE":
        return up in {"OUI", "NON"}

    if up in {"OUI", "NON"}:
        return False

    if expected_class == "DATE":
        # Date manuscrite OCR: on exige un vrai pattern numerique exploitable.
        digits = sum(1 for ch in txt if ch.isdigit())
        return digits >= 4 or bool(re.search(r"\d{1,4}(?:\D+\d{1,4}){2}", txt))

    letters = sum(1 for ch in txt if ch.isalpha())
    digits = sum(1 for ch in txt if ch.isdigit())

    if expected_class in {"IDENTITE", "LIEU", "ETABLISSEMENT", "DIAGNOSTIC"}:
        return letters >= 3 and letters >= digits

    if expected_class == "DUREE":
        if digits >= 1:
            return True
        return any(unit in up for unit in {"JOUR", "JOURS", "MOIS", "AN", "ANS", "HEURE", "HEURES", "SEMAINE", "SEM"})

    return True


def _iter_label_2d_zones(page_number: int, label: str) -> list[tuple[int, int, int, int]]:
    zones: list[tuple[int, int, int, int]] = []
    for class_rules in _get_2d_rules_for_page(page_number).values():
        for x_min, y_min, x_max, y_max, lbl in class_rules:
            if lbl == label:
                zones.append((x_min, y_min, x_max, y_max))
    return zones


def _iter_label_1d_bands(page_number: int, expected_class: str, label: str) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    for y_min, y_max, lbl in _get_rules_for_page(page_number).get(expected_class, []):
        if lbl == label:
            bands.append((y_min, y_max))
    return bands


def _fallback_value_from_predictions(
    label: str,
    expected_class: str,
    predictions: list[dict],
    page_number: int,
    used_prediction_indices: set[int] | None = None,
) -> tuple[str, int | None]:
    """
    Fallback positionnel: recherche un token brut dans la zone du label,
    meme si la classe NER associee est imparfaite.
    """
    candidates: list[tuple[float, str, int]] = []

    zones_2d = _iter_label_2d_zones(page_number, label)
    has_2d_zones = len(zones_2d) > 0
    bands_1d = _iter_label_1d_bands(page_number, expected_class, label)

    for pred_idx, pred in enumerate(predictions):
        if used_prediction_indices and pred_idx in used_prediction_indices:
            continue

        token = str(pred.get("token", "") or "")
        if not _accept_token_for_expected_class(expected_class, token):
            continue

        bb = pred.get("bbox_norm") or []
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        in_2d_zone = False
        zone_distance = 1.0
        for x_min, y_min, x_max, y_max in zones_2d:
            if x_min <= cx <= x_max and y_min <= cy <= y_max:
                in_2d_zone = True
                zx = (x_min + x_max) / 2.0
                zy = (y_min + y_max) / 2.0
                zone_distance = min(1.0, (abs(cx - zx) + abs(cy - zy)) / 1000.0)
                break

        in_1d_band = any(y_min <= cy <= y_max for y_min, y_max in bands_1d)
        if has_2d_zones and not in_2d_zone:
            continue
        if not in_2d_zone and not in_1d_band:
            continue

        yolo_conf = float(pred.get("yolo_conf", 0.0))
        ner_conf = float(pred.get("ner_conf", 0.0))
        base_score = (0.55 * yolo_conf) + (0.35 * ner_conf)
        if in_2d_zone:
            base_score += 0.20
            base_score -= 0.08 * zone_distance
        else:
            base_score += 0.08

        # Favorise les lignes un peu plus riches en contenu pour les champs texte.
        if expected_class != "REPONSE":
            alpha_num = sum(1 for ch in token if ch.isalnum())
            base_score += min(0.08, alpha_num / 200.0)

        candidates.append((base_score, token, pred_idx))

    if not candidates:
        return "", None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def _finalize_label_value(label: str, expected_class: str, text: str, score: float) -> tuple[str, str]:
    raw = text if isinstance(text, str) else ""
    raw_stripped = raw.strip()

    if label == "si_accident_utilise_cim9":
        if not raw_stripped:
            return expected_class, "NON"
        if raw_stripped.upper() in {"OUI", "NON"}:
            return expected_class, raw_stripped.upper()
        if CIM_CODE_RE.match(raw_stripped):
            return expected_class, raw_stripped
        return expected_class, "NON"

    if label == "type_deces":
        return expected_class, _normalize_type_deces_value(raw_stripped)

    if label == "sexe":
        return expected_class, _normalize_sexe_value(raw_stripped)

    if expected_class == "REPONSE":
        normalized = _normalize_yes_no(raw_stripped)
        return expected_class, normalized if normalized.upper() in {"OUI", "NON"} else "NON"

    if not raw_stripped:
        return expected_class, "indefini"

    cleaned = _sanitize_text(raw_stripped)

    if expected_class == "DATE":
        normalized = _normalize_date(cleaned)
        return expected_class, normalized if _is_valid_date(normalized) else "indefini"

    if expected_class == "IDENTITE":
        letters = sum(1 for ch in cleaned if ch.isalpha())
        digits = sum(1 for ch in cleaned if ch.isdigit())
        if letters < 3 or digits > letters:
            return expected_class, "indefini"
        return expected_class, cleaned

    if expected_class in {"LIEU", "ETABLISSEMENT", "DIAGNOSTIC"}:
        letters = sum(1 for ch in cleaned if ch.isalpha())
        digits = sum(1 for ch in cleaned if ch.isdigit())
        if letters < 3 or (digits > 0 and digits > letters):
            return expected_class, "indefini"
        if _looks_like_ocr_noise(label, cleaned):
            return expected_class, "indefini"
        return expected_class, cleaned

    if expected_class == "DUREE":
        if any(unit in cleaned.lower() for unit in ["jour", "jours", "mois", "an", "ans", "heure", "heures", "semaine", "sem"]):
            return expected_class, cleaned
        if any(ch.isdigit() for ch in cleaned):
            return expected_class, cleaned
        return expected_class, "indefini"

    return expected_class, cleaned


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
    def x_center_norm(self) -> int:
        """X centre moyen de toutes les bboxes (coordonnees norm [0-1000])."""
        if not self.bboxes:
            return 0
        return int(sum((bb[0] + bb[2]) / 2 for bb in self.bboxes) / len(self.bboxes))

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
    rejected_low_quality = 0

    for entity in entities:
        entity.real_label, mapping_source = _get_real_label_with_source(
            entity.classe,
            entity.x_center_norm,
            entity.y_center_norm,
            page_number,
        )
        classes_found.add(entity.classe)

        label = entity.real_label
        raw_text = entity.text
        text = raw_text

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

        # Aucun rejet de valeur OCR: on conserve la sortie brute du modele.

        # Si plusieurs valeurs pour un label, garder la plus fiable
        had_existing = label in output
        if (not had_existing) or entity.confidence >= output_conf.get(label, {}).get("score", 0.0):
            is_uncertain = (entity.confidence < 0.55) or (not fmt_valid)
            output[label] = text
            output_conf[label] = {
                "score": round(entity.confidence, 4),
                "uncertain": is_uncertain,
                "classe": entity.classe,
                "mapped_by": mapping_source,
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
        "rejected_low_quality": rejected_low_quality,
        "normalization_changes": normalization_changes,
        "uncertain_fields": uncertain_fields,
    }

    logger.info(f"Mapper : {len(entities)} entites → {max(0, len(output)-2)} champs extraits")
    return output


def map_predictions_page_labels(predictions: list[dict], page_number: int = 1) -> dict:
    """
    Retourne la sortie stricte par page:
    {
      "page_number": N,
      "labels": {
         "label": {"classe": "...", "valeur": "..."}
      }
    }
    """
    specs = PAGE_LABEL_SPECS.get(page_number, [])
    base = map_predictions(predictions, page_number=page_number)
    conf = base.pop("_confidence", {})
    base.pop("_meta", None)

    labels_out: dict[str, dict[str, str]] = {}
    used_fallback_predictions: set[int] = set()

    for label, expected_class in specs:
        raw_value = _extract_candidate_text(label, base)
        mapped_class = str(conf.get(label, {}).get("classe", "") or "").upper()
        if raw_value and mapped_class and mapped_class != expected_class:
            raw_value = ""

        # Ne pas figer une valeur de base tres incertaine si un fallback
        # positionnel peut proposer mieux.
        score = float(conf.get(label, {}).get("score", 0.0))
        if score < 0.52:
            raw_value = ""

        if _is_placeholder_value(raw_value):
            fallback_value, fallback_idx = _fallback_value_from_predictions(
                label=label,
                expected_class=expected_class,
                predictions=predictions,
                page_number=page_number,
                used_prediction_indices=used_fallback_predictions,
            )
            if fallback_value:
                raw_value = fallback_value
                if fallback_idx is not None:
                    used_fallback_predictions.add(fallback_idx)

        final_class, final_value = _finalize_label_value(label, expected_class, raw_value, score)
        if final_value in {"", "[VIDE]", "[ERREUR]", "[TIMEOUT]"}:
            final_value = _default_value_for_class(final_class)
        if final_value == "indefini" and final_class == "REPONSE":
            final_value = "NON"

        labels_out[label] = {
            "classe": final_class,
            "valeur": final_value,
        }

    return {
        "page_number": page_number,
        "labels": labels_out,
    }
