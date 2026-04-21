import torch
import numpy as np
from PIL import Image
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from Frangi_Filter_Pavement_Cracking.soft_frangi_filter2d import SoftFrangiFilter2D
from skimage.filters import apply_hysteresis_threshold


def infer_frangi(pil_image, frangi_filter, device="cpu"):
    img = np.array(pil_image.convert('L')).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        frangi_out = frangi_filter(img_tensor)

    result = -frangi_out.squeeze().cpu().numpy()
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)

    low = np.percentile(result, 90)
    high = np.percentile(result, 97)
    
    binary_pred = apply_hysteresis_threshold(result, low, high).astype(np.uint8)

    return binary_pred

# --- Usage Example ---
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SIGMAS       = [4, 5, 6, 7, 8]
    PERCENTILE   = 92
    ALPHA        = 1
    BETA         = .8
    
    input_img = Image.open("./CRACK500/testcrop/20160222_164141_1281_361.jpg")
    
    frangi_filter = SoftFrangiFilter2D(1, 13, SIGMAS, ALPHA, BETA, DEVICE)
    prediction = infer_frangi(input_img, frangi_filter=frangi_filter, device=DEVICE)
    
    output_pil = Image.fromarray(prediction * 255)
    output_pil.save("Frangi_Pred.png")