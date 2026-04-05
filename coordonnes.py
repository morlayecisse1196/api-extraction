import json
import os

def extraire_annotations(chemin_fichier, labels_a_exclure):
    if not os.path.exists(chemin_fichier):
        print(f"Erreur : Le fichier '{chemin_fichier}' est introuvable.")
        return []

    with open(chemin_fichier, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("Erreur : Le fichier n'est pas un JSON valide.")
            return []
    
    # Correction de l'erreur AttributeError (s'assure que data est itérable)
    if isinstance(data, dict):
        data = [data]
    
    resultats = []

    for item in data:
        # On saute si l'item n'est pas un dictionnaire (évite l'erreur sur 'str')
        if not isinstance(item, dict):
            continue

        image_name = item.get('file_upload', item.get('file_name', 'inconnu'))
        annotations_filtrees = []
        
        # Parcours des annotations
        for ann in item.get('annotations', []):
            for result in ann.get('result', []):
                value = result.get('value', {})
                # Récupération des labels (souvent une liste)
                labels = value.get('rectanglelabels', [])
                
                # LOGIQUE D'EXCLUSION :
                # On ignore l'annotation si elle contient au moins un des labels exclus
                if any(l in labels_a_exclure for l in labels):
                    continue
                
                # Extraction si le label est valide et contient des coordonnées
                if "x" in value and "y" in value:
                    box = {
                        "labels": labels,
                        "x": value.get('x'),
                        "y": value.get('y'),
                        "width": value.get('width'),
                        "height": value.get('height')
                    }
                    annotations_filtrees.append(box)
        
        # On n'ajoute l'image au résultat que si elle a encore des annotations après filtrage
        if annotations_filtrees:
            resultats.append({
                "image": image_name,
                "annotations": annotations_filtrees
            })
    
    return resultats

# --- CONFIGURATION ---
# Ces labels seront strictement ignorés
LABELS_EXCLUS = ["case_vide","case_cochee", "texte_manuscrit", "bruit_rature", "question_imprimee"]
FICHIER_ENTREE = "ANNOTATION\\page4.json"  # Remplacez par votre vrai nom de fichier
FICHIER_SORTIE = "Coordonne_des_labels\\coordonnees_page4.json"

# --- EXÉCUTION ---
try:
    donnees_propres = extraire_annotations(FICHIER_ENTREE, LABELS_EXCLUS)

    # Sauvegarde dans un nouveau fichier JSON
    with open(FICHIER_SORTIE, 'w', encoding='utf-8') as f_out:
        json.dump(donnees_propres, f_out, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"TRAITEMENT TERMINÉ")
    print(f"Images avec annotations valides : {len(donnees_propres)}")
    print(f"Fichier généré : {FICHIER_SORTIE}")
    print("-" * 30)

except Exception as e:
    print(f"Une erreur inattendue est survenue : {e}")