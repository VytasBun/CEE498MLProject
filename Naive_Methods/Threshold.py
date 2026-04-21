from PIL import Image
import numpy as np

def infer_naive_threshold(pil_image, threshold=35):
    gray_img = pil_image.convert('L')
    img_array = np.array(gray_img)

    binary_mask = (img_array < threshold).astype(np.uint8)
    
    return binary_mask

if __name__ == "__main__":
    image_path = ".jpg"
    input_img = Image.open(image_path)

    result_mask = infer_naive_threshold(input_img)
    
    output_pil = Image.fromarray(result_mask * 255)
    output_pil.save("Naive_Threshold_Pred.png")