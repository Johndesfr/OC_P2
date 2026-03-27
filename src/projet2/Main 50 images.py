import os
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
max_images = 3

api_token = ""
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

# Afficher les 3 premiers résultats valides
nb_affiches = 0
for item in batch_seg_results:
    if item["mask"] is not None:
        original_image = Image.open(item["image_path"])
        final_mask = item["mask"]

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Image originale")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(final_mask, cmap="tab20", vmin=0, vmax=17)
        plt.title("Masque segmenté")
        plt.axis("off")

        plt.show()

        nb_affiches += 1
        if nb_affiches == 3:
            break