import os
import math
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import base64
import mimetypes
import io
import time

image_dir = "i:/OC/P2/projet2/src/projet2/content/images/IMG"
max_images = 50

api_token = "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"
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

print("Dossier courant =", os.getcwd())

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Dossier '{image_dir}' créé. Veuillez y ajouter des images .jpg ou .png.")
else:
    print(f"Dossier '{image_dir}' existant.")

# Lister uniquement les fichiers image
extensions_valides = (".jpg", ".jpeg", ".png", ".webp")
image_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith(extensions_valides)
][:max_images]

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter.")

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
    original_image = Image.open(img_path)
    return original_image.size

def decode_base64_mask(base64_string, width, height):
    mask_data = base64.b64decode(base64_string)
    mask_image = Image.open(io.BytesIO(mask_data))
    mask_array = np.array(mask_image)

    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]

    mask_image = Image.fromarray(mask_array).resize((width, height), Image.NEAREST)
    return np.array(mask_image)

def create_masks(results, width, height):
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    for result in results:
        label = result["label"]
        class_id = CLASS_MAPPING.get(label, 0)
        if class_id == 0:
            continue

        mask_array = decode_base64_mask(result["mask"], width, height)
        combined_mask[mask_array > 0] = class_id

    for result in results:
        if result["label"] == "Background":
            mask_array = decode_base64_mask(result["mask"], width, height)
            combined_mask[mask_array > 0] = 0

    return combined_mask

def segment_images_batch(list_of_image_paths):
    batch_segmentations = []

    for image_path in tqdm(list_of_image_paths, desc="Segmentation en cours"):
        try:
            print(f"\nTraitement de l'image : {image_path}")

            # 1) Lire l'image en binaire
            with open(image_path, "rb") as f:
                image_data = f.read()

            # 2) Déterminer le Content-Type
            content_type, _ = mimetypes.guess_type(image_path)
            if content_type is None:
                raise ValueError(f"Impossible de déterminer le type MIME pour {image_path}")

            # 3) Préparer les headers
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": content_type
            }

            # 4) Envoyer la requête POST à l'API
            response = requests.post(API_URL, headers=headers, data=image_data, timeout=60)

            # 5) Vérifier le statut
            response.raise_for_status()

            # 6) Convertir la réponse JSON
            result = response.json()

            # 7) Créer le masque final
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

# Lancer le batch
if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) en batch...")
    batch_seg_results = segment_images_batch(image_paths)
    print("Traitement en batch terminé.")
else:
    batch_seg_results = []
    print("Aucune image à traiter en batch.")



def display_segmented_images_batch(original_image_paths, segmentation_masks):
    """
    Affiche les images originales et leurs masques segmentés.

    Args:
        original_image_paths (list): Liste des chemins des images originales.
        segmentation_masks (list): Liste des résultats de segmentation.
                                   Chaque élément peut être :
                                   - soit un dict {"image_path": ..., "mask": ...}
                                   - soit directement un masque NumPy
    """
    if not original_image_paths or not segmentation_masks:
        print("Aucune image ou aucun masque à afficher.")
        return

    # Construire une liste de paires valides (image, masque)
    valid_pairs = []

    for i in range(min(len(original_image_paths), len(segmentation_masks))):
        image_path = original_image_paths[i]
        seg_item = segmentation_masks[i]

        # Cas 1 : segmentation_masks contient des dictionnaires {"image_path": ..., "mask": ...}
        if isinstance(seg_item, dict):
            mask = seg_item.get("mask", None)
        else:
            # Cas 2 : segmentation_masks contient directement les masques
            mask = seg_item

        if mask is not None:
            valid_pairs.append((image_path, mask))

    if not valid_pairs:
        print("Aucun résultat valide à afficher.")
        return

    n = len(valid_pairs)

    # Nombre de colonnes dans la grille
    cols = 3
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 5, rows * 6))

    for idx, (image_path, mask) in enumerate(valid_pairs):
        # Image originale
        plt.subplot(rows * 2, cols, idx + 1)
        original_image = Image.open(image_path)
        plt.imshow(original_image)
        plt.title("Image originale")
        plt.axis("off")

        # Masque segmenté
        plt.subplot(rows * 2, cols, idx + 1 + (rows * cols))
        plt.imshow(mask, cmap="gray")
        plt.title("Masque segmenté")
        plt.axis("off")

    plt.tight_layout()
    plt.show()