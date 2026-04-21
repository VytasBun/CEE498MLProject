import numpy as np
import cv2
from PIL import Image
from skimage import filters, measure

def infer_otsu_clahe(pil_image, min_area=75):
    img = np.array(pil_image.convert('L'))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    img_clahe = clahe.apply(img)
    img_clahe = img_clahe / 255
    dark_crushed = np.power(1.0 - img_clahe, 10.0)
    img = (dark_crushed * 255).astype(np.uint8)

    thresh_val = filters.threshold_otsu(img)
    binary = img > thresh_val

    labels = measure.label(binary)
    clean_mask = np.zeros_like(binary, dtype=np.uint8)
    
    for region in measure.regionprops(labels):
        if region.area >= min_area:
            for coords in region.coords:
                clean_mask[coords[0], coords[1]] = 1

    return clean_mask

if __name__ == "__main__":
    input_img = Image.open("./CRACK500/testcrop/20160222_164141_1281_361.jpg")
    
    prediction = infer_otsu_clahe(input_img)
    
    output_pil = Image.fromarray(prediction * 255)
    output_pil.save("Otsu_CLAHE_Pred.png")