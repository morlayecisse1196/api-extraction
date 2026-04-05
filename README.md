# API Autopsie Verbale

## Structure

```
api/
├── main.py                    # Point d'entrée FastAPI
├── requirements.txt
├── .env                       # Variables de configuration (optionnel)
├── fichier-cles.json          # Cle de service Google Vision (OCR hybride)
├── core/
│   ├── config.py              # Configuration centralisée
│   └── logger.py              # Logger structuré
├── models_loader/
│   └── loader.py              # Chargement unique des 3 modèles
├── routers/
│   └── extract.py             # Endpoints POST /extract et /extract/batch
├── services/
│   ├── pipeline.py            # Orchestration YOLO → TrOCR → LayoutLMv3
│   └── mapper.py              # Correspondance NER → vrais labels
└── models/                    # Dossier des modèles (à créer)
    ├── best.pt                # Modèle YOLO OBB
    ├── mon_modele_trocr_final_2/   # Modèle TrOCR fine-tuné
    └── layoutlmv3_finetuned/      # Modèle LayoutLMv3 fine-tuné
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration des modèles

Copie tes modèles dans `api/models/` :

```bash
# Depuis ton dossier LAYOUTML/DATA/
cp best.pt                api/models/
cp -r mon_modele_trocr_final_2   api/models/
cp -r layoutlmv3_finetuned       api/models/
```

Ou configure les chemins via `.env` :

```env
YOLO_MODEL_PATH=C:/chemin/complet/vers/best.pt
TROCR_MODEL_PATH=C:/chemin/complet/vers/mon_modele_trocr_final_2
LAYOUTLM_MODEL_PATH=C:/chemin/complet/vers/layoutlmv3_finetuned
```

## Lancement

```bash
cd api/
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

### POST /api/v1/extract
Extrait les données d'une page de formulaire.

**Paramètres :**
- `file` (form-data) : image du formulaire
- `page_number` (query, défaut=1) : numéro de page 1-4
- `ocr_engine` (query, défaut=trocr) : `trocr` ou `google`
- `debug` (query, défaut=false) : inclure les prédictions brutes

**Exemple curl :**
```bash
curl -X POST "http://localhost:8000/api/v1/extract?page_number=1" \
     -F "file=@formulaire_page1.jpg"
```

**Réponse :**
```json
{
  "status": "success",
  "page": 1,
  "filename": "formulaire_page1.jpg",
  "processing_time_s": 3.24,
  "data": {
    "enqueteur":   "Tapo Nickhon",
    "date_visite": "01/03/2026",
    "village":     "Ten Koto",
    "hospitalise": "OUI",
    "_meta": {
      "entities_count": 12,
      "classes_found": ["DATE", "ETABLISSEMENT", "IDENTITE", "REPONSE"],
      "tokens_processed": 18
    }
  }
}
```

### POST /api/v1/extract/batch
Traite les 4 pages d'un formulaire complet en une seule requête.

```bash
curl -X POST "http://localhost:8000/api/v1/extract/batch" \
     -F "files=@page1.jpg" \
     -F "files=@page2.jpg" \
     -F "files=@page3.jpg" \
     -F "files=@page4.jpg"
```

### GET /health
Vérification que l'API est opérationnelle.

## Calibration du mapper

Le fichier `services/mapper.py` contient les plages de positions Y
(coordonnées normalisées [0-1000]) pour associer chaque entité NER
à son vrai label.

Pour calibrer :
1. Lance l'API avec `debug=true`
2. Observe les `bbox_norm` dans `raw_predictions`
3. Ajuste les plages dans `POSITION_RULES` selon les positions réelles
   de chaque champ sur tes formulaires

## Documentation interactive

Disponible sur http://localhost:8000/docs après lancement.

## Presets TrOCR (CPU)

### Preset vitesse max

```env
TROCR_MAX_NEW_TOKENS=24
TROCR_BATCH_SIZE=6
TROCR_MAX_TIME_S=18
MAX_TROCR_ZONES=30
MAP_CHECKBOX_EMPTY_TO_NON=true
```

Usage recommande : priorite a la latence (API plus rapide), au prix d'une legere baisse de precision OCR.

### Preset qualite max

```env
TROCR_MAX_NEW_TOKENS=40
TROCR_BATCH_SIZE=10
TROCR_MAX_TIME_S=35
MAX_TROCR_ZONES=50
MAP_CHECKBOX_EMPTY_TO_NON=true
```

Usage recommande : priorite a la precision OCR, avec un temps de traitement plus long.

### Preset actif dans ce projet

Le fichier [.env](.env) est actuellement regle sur un profil **equilibre CPU** :

```env
TROCR_MAX_NEW_TOKENS=32
TROCR_BATCH_SIZE=8
TROCR_MAX_TIME_S=22
MAX_TROCR_ZONES=35
MAP_CHECKBOX_EMPTY_TO_NON=true
```



Ordre de mise en œuvre recommandé (pratique)

Étapes 1-2
Étapes 4-5
Étapes 7-8
Étapes 3-6
Étapes 9-12

TODO Global (étape par étape)

**1. Baseline et mesures
 Mesurer le temps total actuel sur 10 images représentatives.
 Mesurer par étape: redressement, YOLO zones, TrOCR, LayoutLMv3, mapping.
 Sauvegarder un rapport simple: moyenne, min, max, p95.

**2. Vérifier environnement d’inférence
 Confirmer si PyTorch tourne sur GPU ou CPU au runtime.
 Si CPU: documenter clairement la latence attendue.
 Si GPU dispo: vérifier CUDA, drivers, version torch compatibles.

**3. Stabiliser le redressement de page (coins)
 Ajouter des critères qualité après détection des 4 coins (géométrie valide).
 Si géométrie douteuse: fallback immédiat image originale.
 Ajouter un indicateur dans la réponse debug: rectification appliquée oui/non.
**4. Réduire le bruit de détection YOLO zones
 Filtrer les boîtes texte trop petites / trop fines / hors zone utile.
 Ajuster le seuil de confiance pour réduire les faux positifs.
 Limiter le nombre max de zones envoyées à TrOCR (sécurité perf).

**5. Optimiser TrOCR (gros gain attendu)
 Passer TrOCR en batch (plusieurs crops en un appel).
 Réduire max_new_tokens (ex: 24 ou 32 au lieu de 64).
 Ajouter timeout logique ou garde-fou pour éviter les requêtes extrêmes.
**6. Exploiter correctement les classes YOLO
 Conserver text_line (3) pour OCR.
 Mapper checkbox_checked (0) vers OUI.
 Mapper checkbox_empty (1) vers NON sur les champs concernés.

**7. Rendre le mapping métier robuste
 Utiliser page_number réellement dans le mapping.
 Créer des règles par page (page 1, 2, 3, 4).
 Ajouter validation de format pour certains champs (dates, identifiants).
**8. Ajouter score de confiance champ par champ
 Propager score détection + score NER + heuristique OCR.
 Marquer un champ comme incertain si confiance faible.
 Éviter d’écraser une valeur fiable par une valeur faible en batch.

**9. Nettoyage texte et normalisation
 Normaliser encodage et caractères invalides.
 Nettoyer les artefacts OCR fréquents (séparateurs, doublons, symboles).
 Ajouter post-traitements ciblés sur dates et réponses oui/non.
**10. Mode debug vraiment exploitable
 Retourner timings détaillés par étape.
 Retourner boîtes avant/après redressement.
 Retourner top prédictions utiles pour comprendre les erreurs.

**11. Tests de non-régression
 Créer un petit jeu de test avec vérité terrain.
 Calculer métriques: exact match par champ + temps moyen.
 Comparer avant/après chaque lot de changements.
**12. Durcissement production
 Limiter taille réelle des uploads.
 Restreindre CORS en prod.
 Ajouter logs structurés corrélés par requête.