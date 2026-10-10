
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

def random_resize(x, lo: float = 0.5, hi: float = 1.0, generator=None):
    """Randomly resize tensor down and back up using PyTorch."""
    import torch
    import torch.nn.functional as F
    
    if generator is None:
        scale = torch.rand(1).item() * (hi - lo) + lo
    else:
        scale = torch.rand(1, generator=generator).item() * (hi - lo) + lo
        
    _, _, h, w = x.shape
    new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
    
    down = F.interpolate(x, size=(new_h, new_w), mode='bilinear', antialias=True)
    up = F.interpolate(down, size=(h, w), mode='bilinear', antialias=False)
    return up

def gaussian_blur_t(x, sigma: float):
    """Gaussian blur for tensors (separable depthwise conv2d)."""
    if sigma <= 0:
        return x
    
    import math

    import torch
    import torch.nn.functional as F
    
    ksize = 2 * math.ceil(3 * sigma) + 1
    
    coords = torch.arange(ksize, dtype=torch.float32)
    coords -= (ksize - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    
    kernel_x = g.view(1, 1, 1, ksize).expand(x.shape[1], 1, 1, ksize).to(x.device, dtype=x.dtype)
    kernel_y = g.view(1, 1, ksize, 1).expand(x.shape[1], 1, ksize, 1).to(x.device, dtype=x.dtype)
    
    pad = ksize // 2
    x = F.pad(x, (0, 0, pad, pad), mode='reflect')
    x = F.conv2d(x, kernel_y, groups=x.shape[1])
    x = F.pad(x, (pad, pad, 0, 0), mode='reflect')
    x = F.conv2d(x, kernel_x, groups=x.shape[1])
    
    return x

def jpeg_ste(x, quality: int):
    """JPEG compression with Straight-Through Estimator for gradients."""
    import torch
    
    x_num = x.detach()
    batch_size = x_num.shape[0]
    out_tensors = []
    
    for i in range(batch_size):
        item = x_num[i].cpu().permute(1, 2, 0)
        img_np = (item * 255.0).clamp(0, 255).round().numpy().astype(np.uint8)
        
        jpeg_img = jpeg(img_np, quality)
        
        t = torch.from_numpy(jpeg_img).float().permute(2, 0, 1) / 255.0
        out_tensors.append(t)
        
    x_jpeg = torch.stack(out_tensors).to(x.device)
    return x + (x_jpeg - x).detach()

class EOTSampler:
    """Expectation over Transformation sampler for PGD attacks."""
    def __init__(self, p_resize: float = 0.5, resize_range: tuple[float, float] = (0.5, 1.0),
                 p_blur: float = 0.5, sigma_range: tuple[float, float] = (0.0, 1.5),
                 p_jpeg: float = 0.5, quality_range: tuple[int, int] = (50, 95),
                 seed: int | None = None):
        import torch
        self.p_resize = p_resize
        self.resize_range = resize_range
        self.p_blur = p_blur
        self.sigma_range = sigma_range
        self.p_jpeg = p_jpeg
        self.quality_range = quality_range
        self.generator = torch.Generator(device='cpu')
        if seed is not None:
            self.generator.manual_seed(seed)
            
    def __call__(self, x):
        import torch
        out = x
        
        if torch.rand(1, generator=self.generator).item() < self.p_resize:
            out = random_resize(out, lo=self.resize_range[0], hi=self.resize_range[1], generator=self.generator)
            
        if torch.rand(1, generator=self.generator).item() < self.p_blur:
            sigma = torch.rand(1, generator=self.generator).item() * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
            out = gaussian_blur_t(out, sigma)
            
        if torch.rand(1, generator=self.generator).item() < self.p_jpeg:
            quality = int(torch.rand(1, generator=self.generator).item() * (self.quality_range[1] - self.quality_range[0]) + self.quality_range[0])
            out = jpeg_ste(out, quality)
            
        return out.clamp(0.0, 1.0)
