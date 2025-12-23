import numpy as np
import cv2
from scipy.fftpack import dct

def generate_patch_heatmap(img_tensor, 
                           window_size=50, 
                           stride=10, 
                           sat_power=1.5, 
                           hf_weight=1.0, 
                           blur_kernel=(5, 5)):
    """Generate the suspicion heatmap for adversarial patch presence.
    
    Configurable parameters:
    - window_size: 50 (patch size)
    - stride: 10 (controls heatmap smoothness; possible values: 5, 10, 15, 20)
    - sat_power: 1.5 (saturation amplification; possible values: 1.0, 1.5, 2.0)
    - hf_weight: 1.0 (high-frequency weight; possible values: 0.5, 1.0, 2.0)
    - blur_kernel: (5,5) (final smoothing; None to disable)
    
    Returns:
    - heatmap_norm: numpy uint8 (H, W) – values in [0, 255]
    """
    # Convert to uint8 preserving original colors
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    h, w, _ = img_uint8.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = img_uint8[y:y+window_size, x:x+window_size]
            gray = cv2.cvtColor(window, cv2.COLOR_RGB2GRAY).astype(np.float32)
            saturation = cv2.cvtColor(window, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float32)
            
            var_intensity = np.var(gray)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            mean_grad = np.mean(grad_mag)
            mean_sat = np.mean(saturation) / 255.0
            
            dct_window = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            high_freq_energy = np.sum(np.abs(dct_window[10:, 10:]))
            
            score = var_intensity * mean_grad * (mean_sat ** sat_power) * (high_freq_energy ** hf_weight)
            
            heatmap[y:y+window_size, x:x+window_size] = np.maximum(
                heatmap[y:y+window_size, x:x+window_size], score)
    
    if blur_kernel is not None and blur_kernel != (0, 0):
        heatmap = cv2.GaussianBlur(heatmap, blur_kernel, 0)
    
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return heatmap_norm
