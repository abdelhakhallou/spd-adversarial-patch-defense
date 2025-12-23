
import cv2
import numpy as np
from detection.statistical_detector import generate_patch_heatmap

def mitigate_adversarial_patch(img_tensor, 
                               fixed_thresh=240,
                               opening_kernel=(11, 11),
                               opening_iterations=2,
                               dilation_kernel=(15, 15),
                               dilation_iterations=1,
                               inpaint_radius=5,
                               return_mask=False,
                               return_heatmap=False):
    """Mitige un adversarial patch : détection + suppression + inpainting.
    
    Reproduit exactement l'algo original qui fonctionne parfaitement.
    
    Paramètres :
    - img_tensor : tensor PyTorch (3, 224, 224) – patched (peut être normalisé ou non)
    - fixed_thresh : 240 (valeur optimale pour garder rouge/orange foncé)
    - opening/dilation : morphologie comme l'original
    - inpaint_radius : 5 (TELEA)
    - return_mask/heatmap : pour debug
    
    Retour :
    - cleaned_uint8 : image nettoyée (numpy uint8 RGB)
    - mask (optionnel)
    - heatmap_norm (optionnel)
    """
    # Conversion en uint8 couleurs fidèles
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    # Heatmap via le module detection
    heatmap_norm = generate_patch_heatmap(img_tensor)
    
    # Seuillage fixe
    _, thresh = cv2.threshold(heatmap_norm, fixed_thresh, 255, cv2.THRESH_BINARY)
    
    # Morphologie
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=opening_iterations)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel)
    mask = cv2.dilate(mask, kernel_dilate, iterations=dilation_iterations)
    
    # Inpainting
    cleaned_bgr = cv2.inpaint(img_uint8, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
    
    returns = [cleaned_rgb]
    if return_mask:
        returns.append(mask)
    if return_heatmap:
        returns.append(heatmap_norm)
    
    return tuple(returns) if len(returns) > 1 else returns[0]
