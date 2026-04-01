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

# Construction de la liste des images à traiter
image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(extensions_valides)
][:max_images]

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter.")

# Correspondance nom de classe -> identifiant numérique
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

# Correspondance identifiant numérique -> libellé lisible en français
# Ces libellés serviront pour l'affichage du texte sur le masque
CLASS_NAMES_FR = {
    0: "Fond",
    1: "Chapeau",
    2: "Cheveux",
    3: "Lunettes",
    4: "Haut / veste",
    5: "Jupe",
    6: "Pantalon",
    7: "Robe",
    8: "Ceinture",
    9: "Chaussure gauche",
    10: "Chaussure droite",
    11: "Visage",
    12: "Jambe gauche",
    13: "Jambe droite",
    14: "Bras gauche",
    15: "Bras droit",
    16: "Sac",
    17: "Écharpe"
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
    # Décodage de la chaîne base64 en bytes
    mask_data = base64.b64decode(base64_string)

    # Lecture du masque comme image PIL
    mask_image = Image.open(io.BytesIO(mask_data))

    # Conversion en tableau NumPy
    mask_array = np.array(mask_image)

    # Si le masque a plusieurs canaux (par exemple RGB),
    # on garde uniquement le premier canal
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    # Redimensionnement à la taille de l'image source
    # Image.NEAREST est adapté aux masques de segmentation
    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)

    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Construit un masque unique à partir de la liste des masques renvoyés par l'API.

    Paramètres :
    - results : liste des résultats JSON de l'API
    - width : largeur de l'image d'origine
    - height : hauteur de l'image d'origine

    Retour :
    - masque final combiné sous forme de tableau NumPy
    """
    # Initialisation du masque final avec 0 = fond
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Premier passage : on applique toutes les classes sauf le fond
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)

        # On ignore le fond dans un premier temps
        if class_id == 0:
            continue

        # Décodage du masque de la classe
        mask_array = decode_base64_mask(result["mask"], width, height)

        # Affectation de l'identifiant de classe aux pixels actifs
        combined_mask[mask_array > 0] = class_id

    # Second passage : on réapplique explicitement le fond
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            combined_mask[mask_array > 0] = 0

    return combined_mask

def segment_images_batch(list_of_image_paths):
    """
    Segmente plusieurs images en batch :
    - lecture du fichier image
    - envoi à l'API
    - récupération du JSON
    - création du masque final

    Retour :
    - liste de dictionnaires contenant :
      - image_path
      - mask (ou None en cas d'erreur)
    """
    batch_segmentations = []

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

            # 3) Définition d'une stratégie de retry pour la robustesse HTTP
            retry_strategy = Retry(
                total=5,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["POST"]),
                raise_on_status=False,
                respect_retry_after_header=True,
            )

            # 4) Adaptateur HTTP avec gestion des retries
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # 5) Session HTTP temporaire
            with requests.Session() as session:
                session.mount("https://", adapter)
                session.headers.update({
                    "Authorization": f"Bearer {api_token}",
                    "Accept": "application/json"
                })

                # 6) Envoi de l'image à l'API
                response = session.post(
                    API_URL,
                    headers={"Content-Type": content_type},
                    data=image_data,
                    timeout=(10, 90)
                )

                # 7) Vérification du code HTTP
                response.raise_for_status()

                # 8) Conversion de la réponse en JSON
                result = response.json()

            # 9) Récupération de la taille de l'image d'origine
            width, height = get_image_dimensions(image_path)

            # 10) Construction du masque final
            final_mask = create_masks(result, width, height)

            # 11) Sauvegarde du résultat
            batch_segmentations.append({
                "image_path": image_path,
                "mask": final_mask
            })

        except Exception as e:
            print(f"Une erreur est survenue pour {image_path} : {e}")
            batch_segmentations.append({
                "image_path": image_path,
                "mask": None
            })

    return batch_segmentations

def display_mask_with_labels(original_image, final_mask):
    """
    Affiche côte à côte :
    - l'image originale
    - le masque segmenté

    Et ajoute sur le masque des labels texte centrés
    approximativement sur chaque zone détectée.

    Paramètres :
    - original_image : image PIL originale
    - final_mask : masque NumPy final
    """
    plt.figure(figsize=(12, 5))

    # -----------------------------
    # Sous-figure 1 : image originale
    # -----------------------------
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Image originale")
    plt.axis("off")

    # -----------------------------
    # Sous-figure 2 : masque segmenté
    # -----------------------------
    ax = plt.subplot(1, 2, 2)
    ax.imshow(final_mask, cmap="tab20", vmin=0, vmax=17)
    ax.set_title("Masque segmenté avec labels")
    ax.axis("off")

    # Récupère les classes présentes dans le masque
    # np.unique renvoie tous les IDs distincts
    unique_classes = np.unique(final_mask)

    # On ignore le fond (0)
    unique_classes = unique_classes[unique_classes != 0]

    # Pour chaque classe présente, on calcule une position moyenne
    # afin d'afficher son nom directement sur la zone correspondante
    for class_id in unique_classes:
        # Récupère les coordonnées (ligne, colonne) des pixels appartenant à cette classe
        positions = np.argwhere(final_mask == class_id)

        # Si aucune position n'est trouvée, on passe
        if len(positions) == 0:
            continue

        # Calcul du centre moyen de la zone
        # positions[:, 0] = y
        # positions[:, 1] = x
        y_center = int(np.mean(positions[:, 0]))
        x_center = int(np.mean(positions[:, 1]))

        # Nom à afficher
        label_text = CLASS_NAMES_FR.get(class_id, f"Classe {class_id}")

        # Ajout du texte sur le masque
        # bbox améliore la lisibilité du texte sur fond coloré
        ax.text(
            x_center,
            y_center,
            label_text,
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2)
        )

    plt.show()

# Lancement du batch
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
        # Ouverture de l'image originale
        original_image = Image.open(item["image_path"])

        # Récupération du masque
        final_mask = item["mask"]

        # Affichage du masque avec labels texte
        display_mask_with_labels(original_image, final_mask)

        nb_affiches += 1
        if nb_affiches == 3:
            break