
import torch
from torchvision import transforms
import cv2
import numpy as np
from detection.statistical_detector import generate_patch_heatmap

def defend_against_patch(img_input,
                         target_model="none",       # "imagenet" pour ResNet/AlexNet/etc, "yolo" pour Ultralytics YOLO, "none" pour juste 0-1
                         input_size=(224, 224),     # Taille d'entrée attendue par le modèle (ex: 224x224 ou 640x640)
                         window_size=50,
                         stride=10,
                         sat_power=1.5,
                         hf_weight=1.0,
                         blur_kernel=(5, 5),
                         fixed_thresh=240,
                         opening_kernel=(11, 11),
                         opening_iterations=2,
                         dilation_kernel=(15, 15),
                         dilation_iterations=1,
                         inpaint_radius=5):
    """
    Pipeline complet de défense contre les adversarial patches - VERSION PLUG & PLAY.
    
    Entrée :
        - img_input : soit numpy array uint8 RGB (H, W, 3) - image brute de caméra
                  soit torch Tensor (3, H, W) - déjà préprocessé (comme dans le PoC original)
    
    Paramètres importants :
        - target_model : choisit la normalisation de sortie
            "imagenet" -> applique la normalisation standard ImageNet
            "yolo"     -> pas de normalisation supplémentaire (YOLO attend 0-1 ou uint8)
            "none"     -> sortie en 0-1 simple
        - input_size : taille à laquelle redimensionner l'image (obligatoire pour cohérence)
    
    Sortie :
        torch Tensor prêt à être donné directement au modèle choisi.
    """
    # ===================================================================
    # 1. Gestion de l'entrée : image brute ou tensor déjà traité ?
    # ===================================================================
    if isinstance(img_input, np.ndarray):
        # Image brute de caméra : numpy uint8 RGB 0-255
        img_rgb = img_input.copy()
        # Redimensionner si besoin
        if img_rgb.shape[:2] != input_size:
            img_rgb = cv2.resize(img_rgb, input_size[::-1])  # OpenCV attend (largeur, hauteur)
        # Convertir en tensor temporaire pour la heatmap
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        already_normalized = False
        img_uint8 = img_rgb  # déjà en uint8 RGB
    else:
        # Entrée déjà un tensor (comme dans le PoC)
        img_tensor = img_input
        already_normalized = img_tensor.min() < 0 or img_tensor.max() > 1
        
        # Ramener temporairement en uint8 RGB avec couleurs naturelles
        if already_normalized:
            # Dé-normalisation ImageNet temporaire
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            temp = img_tensor * std + mean
            temp = torch.clamp(temp, 0, 1)
            img_uint8 = (temp.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            img_uint8 = np.clip(img_tensor.cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        
        # Redimensionner si besoin (rare dans le PoC, mais sécurité)
        if img_uint8.shape[:2] != input_size:
            img_uint8 = cv2.resize(img_uint8, input_size[::-1])
            # Mettre à jour le tensor aussi
            img_tensor = torch.from_numpy(img_uint8.transpose(2, 0, 1)).float() / 255.0

    # ===================================================================
    # 2. Détection du patch -> heatmap
    # ===================================================================
    heatmap_norm = generate_patch_heatmap(img_tensor, window_size, stride, sat_power, hf_weight, blur_kernel)

    # ===================================================================
    # 3. Création du masque
    # ===================================================================
    _, thresh = cv2.threshold(heatmap_norm, fixed_thresh, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=opening_iterations)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel)
    mask = cv2.dilate(mask, kernel_dilate, iterations=dilation_iterations)

    # ===================================================================
    # 4. Inpainting (remplissage naturel de la zone)
    # ===================================================================
    # OpenCV attend du BGR -> conversion explicite pour éviter les dominantes bleues
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cleaned_bgr = cv2.inpaint(img_bgr, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

    # ===================================================================
    # 5. Préparation de la sortie selon le modèle cible
    # ===================================================================
    cleaned_tensor = torch.from_numpy(cleaned_rgb.transpose(2, 0, 1)).float() / 255.0

    if target_model == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        cleaned_tensor = normalize(cleaned_tensor)
    # Pour "yolo" ou "none" -> on laisse en 0-1, rien à faire

    return cleaned_tensor
