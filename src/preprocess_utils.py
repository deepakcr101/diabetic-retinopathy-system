import cv2
import numpy as np

def crop_image_from_gray(img, tol=7):
    """
    Crops the black borders from a fundus image.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0): # Image is too dark
            return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img

def apply_clahe(img):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the green channel.
    The green channel contains the most details for DR lesions.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    
    # Merge and convert back to RGB
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def preprocess_image(image_path, target_size=512):
    """
    Full preprocessing pipeline: Read -> Crop -> Resize -> CLAHE.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Crop black borders
    img = crop_image_from_gray(img)
    
    # 2. Resize
    img = cv2.resize(img, (target_size, target_size))
    
    # 3. Apply CLAHE
    img = apply_clahe(img)
    
    return img
