import os
import re
import requests
from PIL import Image
import numpy as np
from tqdm import tqdm
import base64
import mimetypes
import io
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# =========================
# PARAMÈTRES
# =========================

# Dossier contenant les images à segmenter
image_dir = "i:/OC/P2/projet2/src/projet2/content/images/IMG"

# Dossier contenant les masques GT
gt_dir = "i:/OC/P2/projet2/src/projet2/content/images/Mask"

# Nombre maximum d'images à traiter
max_images = 50

# Token d'authentification Hugging Face
load_dotenv()
api_token = os.getenv("HF_API_TOKEN")

if not api_token:
    raise ValueError("HF_API_TOKEN introuvable. Ajoutez-le dans le fichier .env.")

# URL du modèle de segmentation
API_URL = "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"

print("Dossier courant =", os.getcwd())

# Vérifie si les dossiers existent
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"Dossier images introuvable : {image_dir}")

if not os.path.exists(gt_dir):
    raise FileNotFoundError(f"Dossier GT introuvable : {gt_dir}")

# Extensions acceptées
extensions_valides = (".jpg", ".jpeg", ".png", ".webp")

# =========================
# MAPPINGS ET CLASSES
# =========================

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

# Fusion des chaussures gauche/droite
SHOES_CLASS_ID = 9

# Classes finales après fusion de 9 et 10
FINAL_CLASS_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]

CLASS_NAMES_FR = {
    0: "Arrière-plan",
    1: "Chapeau",
    2: "Cheveux",
    3: "Lunettes de soleil",
    4: "Haut (vêtement)",
    5: "Jupe",
    6: "Pantalon",
    7: "Robe",
    8: "Ceinture",
    9: "Chaussures",
    11: "Visage",
    12: "Jambe gauche",
    13: "Jambe droite",
    14: "Bras gauche",
    15: "Bras droit",
    16: "Sac",
    17: "Écharpe"
}

# Mapping vers des indices compacts [0..N-1]
COMPACT_CLASS_MAP = {class_id: idx for idx, class_id in enumerate(FINAL_CLASS_IDS)}
COMPACT_TO_ORIGINAL = {idx: class_id for class_id, idx in COMPACT_CLASS_MAP.items()}
NUM_CLASSES = len(FINAL_CLASS_IDS)

# =========================
# FONCTIONS UTILITAIRES
# =========================

def natural_sort_key(path):
    """
    Clé de tri naturel pour classer correctement :
    image_0, image_1, image_2, ..., image_10
    au lieu de :
    image_0, image_1, image_10, image_2
    """
    filename = os.path.basename(path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r'(\d+)', filename)
    ]

def normalize_class_id(class_id):
    """
    Fusionne Left-shoe (9) et Right-shoe (10) en une seule classe : Chaussures (9).
    """
    if class_id in (9, 10):
        return SHOES_CLASS_ID
    return class_id

def get_image_dimensions(img_path):
    """
    Ouvre l'image et retourne ses dimensions (largeur, hauteur).
    """
    with Image.open(img_path) as original_image:
        return original_image.size

def decode_base64_mask(base64_string, width, height):
    """
    Décode un masque base64 et le redimensionne à la taille de l'image d'origine.
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
    Construit un masque final robuste à partir des résultats API.

    Stratégie :
    - décodage de tous les masques
    - utilisation du score si disponible
    - résolution des conflits par score puis priorité
    """
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    score_map = np.full((height, width), -1.0, dtype=np.float32)

    class_priority = {
        9: 100,   # Chaussures
        16: 80,   # Sac
        8: 70,    # Ceinture
        4: 60,    # Haut
        5: 60,    # Jupe
        6: 60,    # Pantalon
        7: 60,    # Robe
        14: 50,   # Bras gauche
        15: 50,   # Bras droit
        12: 50,   # Jambe gauche
        13: 50,   # Jambe droite
        11: 50,   # Visage
        1: 40,    # Chapeau
        2: 40,    # Cheveux
        3: 40,    # Lunettes
        17: 40,   # Écharpe
        0: 0
    }

    decoded_items = []

    for result in results:
        label = result["label"]
        raw_class_id = CLASS_MAPPING.get(label, 0)
        class_id = normalize_class_id(raw_class_id)

        mask_array = decode_base64_mask(result["mask"], width, height)
        score = float(result.get("score", 1.0))

        decoded_items.append({
            "class_id": class_id,
            "mask": mask_array > 0,
            "score": score,
            "priority": class_priority.get(class_id, 1)
        })

    foreground_items = [item for item in decoded_items if item["class_id"] != 0]

    foreground_items.sort(
        key=lambda x: (x["score"], x["priority"]),
        reverse=True
    )

    current_priority_map = np.zeros((height, width), dtype=np.int16)

    for item in foreground_items:
        class_id = item["class_id"]
        mask_bool = item["mask"]
        score = item["score"]
        priority = item["priority"]

        replace_better_score = mask_bool & (score > score_map)

        replace_equal_score = (
            mask_bool
            & np.isclose(score, score_map)
            & (priority > current_priority_map)
        )

        replace = replace_better_score | replace_equal_score

        combined_mask[replace] = class_id
        score_map[replace] = score
        current_priority_map[replace] = priority

    return combined_mask

def build_gt_path(image_index):
    """
    Construit le chemin du masque GT à partir de l'index de l'image.
    Exemple :
    image 0 -> mask_0.png
    image 1 -> mask_1.png
    """
    return os.path.join(gt_dir, f"mask_{image_index}.png")

def load_gt_mask(gt_path, target_width, target_height):
    """
    Charge le masque GT, le redimensionne à la taille cible si nécessaire,
    puis normalise les classes.

    Hypothèse :
    le masque GT est un masque indexé où chaque pixel contient directement
    l'ID de classe.
    """
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Masque GT introuvable : {gt_path}")

    gt_image = Image.open(gt_path)
    gt_array = np.array(gt_image)

    # Si le masque est sur 3 canaux mais contient en réalité les mêmes IDs,
    # on prend le premier canal.
    if len(gt_array.shape) == 3:
        gt_array = gt_array[:, :, 0]

    # Redimensionnement si nécessaire
    if gt_array.shape != (target_height, target_width):
        gt_array = np.array(
            Image.fromarray(gt_array).resize((target_width, target_height), Image.NEAREST)
        )

    # Normalisation des classes (fusion des chaussures)
    gt_normalized = np.vectorize(normalize_class_id)(gt_array).astype(np.uint8)

    return gt_normalized

def remap_to_compact(mask):
    """
    Remappe un masque avec IDs originaux vers des IDs compacts [0..NUM_CLASSES-1].
    Les classes inconnues sont ignorées avec 255.
    """
    remapped = np.full(mask.shape, 255, dtype=np.uint8)

    for original_class_id, compact_idx in COMPACT_CLASS_MAP.items():
        remapped[mask == original_class_id] = compact_idx

    return remapped

def compute_confusion_matrix(pred, gt, num_classes, ignore_index=255):
    """
    Calcule la matrice de confusion pour une image.
    Lignes = GT, colonnes = prédiction.
    """
    valid_mask = (gt != ignore_index) & (pred != ignore_index)
    gt_valid = gt[valid_mask].astype(np.int64)
    pred_valid = pred[valid_mask].astype(np.int64)

    hist = np.bincount(
        num_classes * gt_valid + pred_valid,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return hist

def compute_iou_from_confusion(conf_matrix):
    """
    Calcule l'IoU par classe à partir de la matrice de confusion globale.
    """
    iou_dict = {}

    for compact_idx in range(NUM_CLASSES):
        original_class_id = COMPACT_TO_ORIGINAL[compact_idx]

        tp = conf_matrix[compact_idx, compact_idx]
        fp = conf_matrix[:, compact_idx].sum() - tp
        fn = conf_matrix[compact_idx, :].sum() - tp

        denom = tp + fp + fn

        if denom == 0:
            iou = np.nan
        else:
            iou = tp / denom

        iou_dict[original_class_id] = iou

    return iou_dict

def segment_images_and_compute_iou(list_of_image_paths):
    """
    Segmente les images, charge les GT correspondants, accumule la matrice
    de confusion globale et calcule l'IoU final.
    """
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    processed_images = 0
    skipped_images = 0

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

        for image_index, image_path in enumerate(tqdm(list_of_image_paths, desc="Évaluation IoU en cours")):
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

                # Masque prédit
                pred_mask = create_masks(result, width, height)

                # Masque GT associé à l'index de l'image
                gt_path = build_gt_path(image_index)
                gt_mask = load_gt_mask(gt_path, width, height)

                # Remapping vers classes compactes
                pred_compact = remap_to_compact(pred_mask)
                gt_compact = remap_to_compact(gt_mask)

                # Accumulation matrice de confusion
                conf_matrix += compute_confusion_matrix(pred_compact, gt_compact, NUM_CLASSES)

                processed_images += 1

                print(f"Image index {image_index} : {os.path.basename(image_path)}")
                print("GT utilisé :", gt_path)
                print("Classes prédites présentes :", np.unique(pred_mask))
                print("Classes GT présentes      :", np.unique(gt_mask))

            except Exception as e:
                print(f"Une erreur est survenue pour {image_path} : {e}")
                skipped_images += 1

    iou_per_class = compute_iou_from_confusion(conf_matrix)

    miou_all = np.nanmean(list(iou_per_class.values()))
    miou_no_bg = np.nanmean([
        iou for class_id, iou in iou_per_class.items()
        if class_id != 0
    ])

    return {
        "conf_matrix": conf_matrix,
        "iou_per_class": iou_per_class,
        "miou_all": miou_all,
        "miou_no_bg": miou_no_bg,
        "processed_images": processed_images,
        "skipped_images": skipped_images
    }

# =========================
# LISTE DES IMAGES
# =========================

image_paths = sorted(
    [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(extensions_valides)
    ],
    key=natural_sort_key
)[:max_images]

if not image_paths:
    print(f"Aucune image trouvée dans '{image_dir}'. Veuillez y ajouter des images.")
else:
    print(f"{len(image_paths)} image(s) à traiter.")

    print("\nOrdre des images sélectionnées :")
    for idx, path in enumerate(image_paths):
        print(f"{idx} -> {os.path.basename(path)} -> mask_{idx}.png")

# =========================
# LANCEMENT
# =========================

if image_paths:
    print(f"\nTraitement de {len(image_paths)} image(s) pour calculer l'IoU...")
    results = segment_images_and_compute_iou(image_paths)

    print("\n=========================")
    print("RÉSULTATS FINAUX")
    print("=========================")
    print(f"Images traitées : {results['processed_images']}")
    print(f"Images ignorées : {results['skipped_images']}")

    print("\nIoU par classe :")
    for class_id in FINAL_CLASS_IDS:
        class_name = CLASS_NAMES_FR[class_id]
        iou = results["iou_per_class"][class_id]

        if np.isnan(iou):
            print(f"- {class_name:20s} : absent du jeu évalué")
        else:
            print(f"- {class_name:20s} : {iou:.4f}")

    print(f"\nmIoU (avec background) : {results['miou_all']:.4f}")
    print(f"mIoU (sans background) : {results['miou_no_bg']:.4f}")

    print("\nMatrice de confusion globale :")
    print(results["conf_matrix"])

else:
    print("Aucune image à traiter.")