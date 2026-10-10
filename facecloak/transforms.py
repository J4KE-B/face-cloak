
import numpy as np


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    """Apply JPEG compression to an RGB uint8 image."""
    import cv2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

def resize_down_up(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize image down and back up to original size."""
    import cv2
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    if new_w == 0 or new_h == 0:
        return img.copy()
    down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur to an RGB uint8 image."""
    if sigma <= 0:
        return img.copy()
    import math

    import cv2
    ksize = 2 * math.ceil(3 * sigma) + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def median(img: np.ndarray, k: int) -> np.ndarray:
    """Apply median blur."""
    import cv2
    return cv2.medianBlur(img, k)

def bilateral(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filter."""
    import cv2
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

ROBUSTNESS = {
    "none": lambda x: x.copy(),
    "jpeg90": lambda x: jpeg(x, 90),
    "jpeg75": lambda x: jpeg(x, 75),
    "jpeg50": lambda x: jpeg(x, 50),
    "resize0.75": lambda x: resize_down_up(x, 0.75),
    "resize0.5": lambda x: resize_down_up(x, 0.5),
    "blur1": lambda x: gaussian_blur(x, 1.0),
    "blur2": lambda x: gaussian_blur(x, 2.0),
    "median3": lambda x: median(x, 3),
    "bilateral": lambda x: bilateral(x)
}
