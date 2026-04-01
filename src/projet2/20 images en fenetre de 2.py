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
from dotenv import load_dotenv

# Dossier contenant les images à segmenter
image_dir = "i:/OC/P2/projet2/src/projet2/content/images/IMG"

# Nombre maximum d'images à traiter
max_images = 20

# Nombre d'images affichées par fenêtre
images_per_window = 2

# Token d'authentification Hugging Face
load_dotenv()
api_token = os.getenv("HF_API_TOKEN")

if not api_token:
    raise ValueError("HF_API_TOKEN introuvable. Ajoutez-le dans le fichier .env.")

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
    """
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)

    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    """
    Construit un masque unique à partir de la liste des masques renvoyés par l'API.
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # Premier passage : toutes les classes sauf le fond
    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)

        if class_id == 0:
            continue

        mask_array = decode_base64_mask(result["mask"], width, height)
        combined_mask[mask_array > 0] = class_id

    # Second passage : réapplication explicite du fond
    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            combined_mask[mask_array > 0] = 0

    return combined_mask

def segment_images_batch(list_of_image_paths):
    """
    Segmente plusieurs images en batch.
    Retour :
    - liste de dictionnaires contenant :
      - image_path
      - mask (ou None en cas d'erreur)
    """
    batch_segmentations = []

    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)

    with requests.Session() as session:
        session.mount("https://", adapter)
        session.headers.update({
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json"
        })

        for image_path in tqdm(list_of_image_paths, desc="Segmentation en cours"):
            try:
                print(f"\nTraitement de l'image : {image_path}")

                with open(image_path, "rb") as f:
                    image_data = f.read()

                content_type, _ = mimetypes.guess_type(image_path)
                if content_type is None:
                    raise ValueError(f"Impossible de déterminer le type MIME pour {image_path}")

                response = session.post(
                    API_URL,
                    headers={"Content-Type": content_type},
                    data=image_data,
                    timeout=(10, 90)
                )

                response.raise_for_status()
                result = response.json()

                width, height = get_image_dimensions(image_path)
                final_mask = create_masks(result, width, height)

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

def add_labels_on_axis(ax, final_mask):
    """
    Ajoute les labels texte sur le masque coloré.
    """
    unique_classes = np.unique(final_mask)
    unique_classes = unique_classes[unique_classes != 0]

    for class_id in unique_classes:
        positions = np.argwhere(final_mask == class_id)

        if len(positions) == 0:
            continue

        y_center = int(np.mean(positions[:, 0]))
        x_center = int(np.mean(positions[:, 1]))

        label_text = CLASS_NAMES_FR.get(class_id, f"Classe {class_id}")

        ax.text(
            x_center,
            y_center,
            label_text,
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1)
        )

def display_results_paginated(batch_seg_results, images_per_window=5):
    """
    Affiche les résultats par groupes de 5 images par fenêtre.
    Chaque image occupe une ligne avec 3 colonnes :
    - image originale
    - masque brut
    - masque coloré et labellisé
    """
    valid_results = [item for item in batch_seg_results if item["mask"] is not None]

    if not valid_results:
        print("Aucun résultat valide à afficher.")
        return

    total_images = len(valid_results)

    for start_idx in range(0, total_images, images_per_window):
        end_idx = min(start_idx + images_per_window, total_images)
        chunk = valid_results[start_idx:end_idx]
        n = len(chunk)

        fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(18, 4 * n))

        if n == 1:
            axes = np.array([axes])

        for i, item in enumerate(chunk):
            original_image = Image.open(item["image_path"]).convert("RGB")
            final_mask = item["mask"]

            image_number = start_idx + i + 1
            file_name = os.path.basename(item["image_path"])

            # Colonne 1 : image originale
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Image {image_number} - Originale\n{file_name}", fontsize=9)
            axes[i, 0].axis("off")

            # Colonne 2 : masque brut
            axes[i, 1].imshow(final_mask, cmap="gray")
            axes[i, 1].set_title(f"Image {image_number} - Masque brut", fontsize=9)
            axes[i, 1].axis("off")

            # Colonne 3 : masque coloré + labels
            axes[i, 2].imshow(final_mask, cmap="tab20", vmin=0, vmax=17)
            axes[i, 2].set_title(f"Image {image_number} - Masque coloré", fontsize=9)
            axes[i, 2].axis("off")

            add_labels_on_axis(axes[i, 2], final_mask)

        fig.suptitle(
            f"Résultats de segmentation - Images {start_idx + 1} à {end_idx}",
            fontsize=14
        )

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()

# Lancement du batch
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = segment_images_batch(image_paths)
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")

# Affichage paginé : 5 images par fenêtre
display_results_paginated(batch_seg_results, images_per_window=2)