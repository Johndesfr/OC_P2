import os
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import base64
import mimetypes
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Dossier contenant les images à segmenter
image_dir = "i:/OC/P2/projet2/src/projet2/content/images/IMG"

# Nombre maximum d'images à traiter
max_images = 3

# Token d'authentification Hugging Face
api_token = "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"

# URL du modèle de segmentation
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

# Affiche le dossier courant d'exécution du script
print("Dossier courant =", os.getcwd())

# Vérifie si le dossier d'images existe, sinon le crée
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Dossier '{image_dir}' créé. Veuillez y ajouter des images .jpg ou .png.")
else:
    print(f"Dossier '{image_dir}' existant.")

# Extensions d'images acceptées
extensions_valides = (".jpg", ".jpeg", ".png", ".webp")

# Construit la liste des chemins des images à traiter
# On filtre uniquement les fichiers ayant une extension valide
# Puis on limite à max_images
image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(extensions_valides)
][:max_images]

# Vérifie si des images ont été trouvées
if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter.")

# Dictionnaire de correspondance entre les labels du modèle
# et les identifiants numériques utilisés dans le masque final
CLASS_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17
}

def get_image_dimensions(img_path):
    """
    Ouvre l'image et retourne ses dimensions d'origine
    sous la forme (largeur, hauteur).
    """
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Décode un masque encodé en base64 renvoyé par l'API,
    puis le redimensionne aux dimensions de l'image d'origine.

    Paramètres :
    - base64_string : chaîne base64 contenant l'image du masque
    - width : largeur cible
    - height : hauteur cible

    Retour :
    - masque sous forme de tableau NumPy 2D
    """
    # Décodage base64 -> bytes
    mask_data = base64.b64decode(base64_string)

    # Lecture du masque en image PIL depuis les bytes
    mask_image = Image.open(io.BytesIO(mask_data))

    # Conversion en tableau NumPy
    mask_array = np.array(mask_image)

    # Si le masque possède plusieurs canaux (ex. RGB),
    # on ne conserve qu'un seul canal
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    # Redimensionnement à la taille de l'image d'origine
    # avec interpolation nearest pour conserver des classes entières
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)

    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Construit un masque de segmentation final unique
    à partir de la liste des résultats renvoyés par l'API.

    Paramètres :
    - results : liste des objets retournés par l'API
    - width : largeur de l'image d'origine
    - height : hauteur de l'image d'origine

    Retour :
    - masque combiné sous forme de tableau NumPy
    """
    # Initialisation du masque final avec 0 = arrière-plan
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Premier passage : on applique toutes les classes sauf le fond
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)

        # Si la classe n'est pas reconnue ou correspond au fond, on ignore
        if class_id == 0:
            continue

        # Décodage du masque de la classe
        mask_array = decode_base64_mask(result["mask"], width, height)

        # On place l'identifiant de classe sur les pixels actifs
        combined_mask[mask_array > 0] = class_id

    # Deuxième passage : traitement spécifique du background
    # pour remettre à 0 les zones d'arrière-plan si nécessaire
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            combined_mask[mask_array > 0] = 0

    return combined_mask

def segment_images_batch(list_of_image_paths):
    """
    Traite une liste d'images en batch :
    - lecture binaire de chaque image
    - envoi à l'API
    - récupération du JSON
    - création du masque final

    Retour :
    - liste de dictionnaires contenant :
      - image_path
      - mask (ou None en cas d'erreur)
    """
    batch_segmentations = []

    # Boucle sur toutes les images avec barre de progression
    for image_path in tqdm(list_of_image_paths, desc="Segmentation en cours"):
        try:
            print(f"\nTraitement de l'image : {image_path}")

            # 1) Lire l'image en binaire
            with open(image_path, "rb") as f:
                image_data = f.read()

            # 2) Déterminer automatiquement le type MIME
            content_type, _ = mimetypes.guess_type(image_path)
            if content_type is None:
                raise ValueError(f"Impossible de déterminer le type MIME pour {image_path}")

            # 3) Définir une stratégie de retry pour rendre les requêtes plus robustes
            retry_strategy = Retry(
                total=5,  # nombre total de tentatives
                backoff_factor=1,  # temporisation progressive entre les retries
                status_forcelist=[429, 500, 502, 503, 504],  # codes HTTP à retenter
                allowed_methods=frozenset(["POST"]),  # on autorise aussi les retries sur POST
                raise_on_status=False,
                respect_retry_after_header=True,
            )

            # 4) Création de l'adaptateur HTTP avec la stratégie de retry
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # 5) Création d'une session HTTP
            with requests.Session() as session:
                # On monte l'adaptateur sur HTTPS
                session.mount("https://", adapter)

                # Headers communs de la session
                session.headers.update({
                    "Authorization": f"Bearer {api_token}",
                    "Accept": "application/json"
                })

                # 6) Envoi de l'image à l'API via POST
                response = session.post(
                    API_URL,
                    headers={"Content-Type": content_type},
                    data=image_data,
                    timeout=(10, 90)  # timeout connexion / lecture
                )

                # 7) Déclenche une exception si code HTTP 4xx ou 5xx
                response.raise_for_status()

                # 8) Conversion de la réponse en JSON
                result = response.json()

            # 9) Récupération des dimensions de l'image d'origine
            width, height = get_image_dimensions(image_path)

            # 10) Création du masque final combiné
            final_mask = create_masks(result, width, height)

            # 11) Sauvegarde du résultat dans la liste finale
            batch_segmentations.append({
                "image_path": image_path,
                "mask": final_mask
            })

        except Exception as e:
            # En cas d'erreur, on logge et on stocke un masque à None
            print(f"Une erreur est survenue pour {image_path} : {e}")
            batch_segmentations.append({
                "image_path": image_path,
                "mask": None
            })

    return batch_segmentations

# Lancer le traitement batch si des images ont été trouvées
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = segment_images_batch(image_paths)
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")

# Affichage des 3 premiers résultats valides
nb_affiches = 0

for item in batch_seg_results:
    if item["mask"] is not None:
        # Ouvre l'image originale
        original_image = Image.open(item["image_path"])

        # Récupère le masque calculé
        final_mask = item["mask"]

        # Crée une figure d'affichage
        plt.figure(figsize=(12, 5))

        # Sous-figure 1 : image originale
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Image originale")
        plt.axis("off")

        # Sous-figure 2 : masque segmenté
        plt.subplot(1, 2, 2)
        plt.imshow(final_mask, cmap="tab20", vmin=0, vmax=17)
        plt.title("Masque segmenté")
        plt.axis("off")

        # Affiche les deux images
        plt.show()

        nb_affiches += 1
        if nb_affiches == 3:
            break