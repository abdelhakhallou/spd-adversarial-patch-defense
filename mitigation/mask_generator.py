
import cv2
import numpy as np

def get_patch_mask(heatmap_norm, 
                   fixed_thresh=100,
                   opening_kernel=(11, 11),
                   opening_iterations=2,
                   dilation_kernel=(15, 15),
                   dilation_iterations=1,
                   keep_largest=True):
    """Génère un masque binaire précis à partir de la heatmap.
    
    Paramètres configurables (valeurs optimisées) :
    - fixed_thresh : 240 (garde le rouge/orange foncé, valeurs possibles : 200-250)
    - opening_kernel / iterations : nettoyage du bruit
    - dilation_kernel / iterations : connexion du patch
    - keep_largest : garde la plus grande composante connexe (utile contre faux positifs)
    
    Retour :
    - mask : uint8 (H, W) – 0 ou 255
    """
    # Seuillage fixe
    _, thresh = cv2.threshold(heatmap_norm, fixed_thresh, 255, cv2.THRESH_BINARY)
    
    # Opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=opening_iterations)
    
    # Dilation
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel)
    mask = cv2.dilate(mask, kernel_dilate, iterations=dilation_iterations)
    
    # Option : garder la plus grande composante
    if keep_largest:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            # Ignorer le fond (label 0)
            largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            mask = np.zeros_like(mask)
            mask[labels == largest_label] = 255
    
    return mask
