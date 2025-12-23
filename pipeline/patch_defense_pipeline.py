
import torch
from torchvision import transforms
import cv2
import numpy as np
from detection.statistical_detector import generate_patch_heatmap

def defend_against_patch(img_input,
                         target_model="imagenet",   # "imagenet", "yolo", or "none"
                         input_size=(224, 224),     # Target input size
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
    """Complete SPD pipeline – faithful to original working version with perfect color preservation.
    
    Input:
        - img_input: numpy uint8 RGB array (H, W, 3) or torch Tensor (3, H, W) [0-1 or normalized].
    Output:
        - torch Tensor ready for the target model.
    """
    # 1. Convert input to uint8 RGB (preserving original colors)
    if isinstance(img_input, np.ndarray):
        img_uint8 = img_input.copy()  # Already uint8 RGB
    else:
        img_tensor = img_input
        if img_tensor.min() < 0 or img_tensor.max() > 1:
            # Denormalize ImageNet if needed
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            temp = img_tensor * std + mean
            temp = torch.clamp(temp, 0, 1)
        else:
            temp = img_tensor
        img_uint8 = np.clip(temp.cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)

    # Resize if necessary
    if img_uint8.shape[:2] != input_size:
        img_uint8 = cv2.resize(img_uint8, input_size[::-1])  # (width, height)

    # 2. Generate suspicion heatmap (using resized tensor)
    img_tensor_resized = torch.from_numpy(img_uint8.transpose(2, 0, 1)).float() / 255.0
    heatmap_norm = generate_patch_heatmap(img_tensor_resized, window_size, stride, sat_power, hf_weight, blur_kernel)

    # 3. Mask refinement
    _, thresh = cv2.threshold(heatmap_norm, fixed_thresh, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=opening_iterations)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel)
    mask = cv2.dilate(mask, kernel_dilate, iterations=dilation_iterations)

    # 4. Inpainting – EXACT ORIGINAL COLOR HANDLING (no bug)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)        # RGB → BGR for OpenCV
    cleaned_bgr = cv2.inpaint(img_bgr, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)  # BGR → RGB back

    # 5. Prepare output tensor
    cleaned_tensor = torch.from_numpy(cleaned_rgb.transpose(2, 0, 1)).float() / 255.0

    if target_model == "imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        cleaned_tensor = normalize(cleaned_tensor)
    # "yolo" or "none": return [0,1] tensor

    return cleaned_tensor
