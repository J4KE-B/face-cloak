import math

import numpy as np


def cosine(a, b):
    """Compute cosine similarity between two arrays (1D or batched N,D)."""
    a = np.asarray(a)
    b = np.asarray(b)
    squeeze = False
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        squeeze = True
        
    a_norm = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=-1, keepdims=True)
    
    sim = np.sum(a_norm * b_norm, axis=-1)
    if squeeze:
        return sim[0]
    return sim

def psnr(img_a: np.ndarray, img_b: np.ndarray, data_range: int = 255) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two uint8 RGB images."""
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two uint8 RGB images."""
    from skimage.metrics import structural_similarity
    return structural_similarity(img_a, img_b, channel_axis=2, data_range=255)

class LPIPSMetric:
    """LPIPS perceptual distance metric."""
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self._loss_fn = None

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute LPIPS distance. Inputs are uint8 HxWx3 images."""
        if self._loss_fn is None:
            import lpips
            self._loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
        
        import torch
        
        def preprocess(img):
            tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
            return (tensor / 255.0) * 2.0 - 1.0

        t_a = preprocess(a).to(self.device)
        t_b = preprocess(b).to(self.device)
        
        with torch.no_grad():
            return self._loss_fn(t_a, t_b).item()

def protection_success_rate(sims: np.ndarray, tau: float) -> tuple[float, int]:
    """Calculate protection success rate (fraction of non-NaN sims < tau)."""
    sims = np.asarray(sims)
    valid = sims[~np.isnan(sims)]
    if len(valid) == 0:
        return np.nan, 0
    success = np.sum(valid < tau)
    return float(success / len(valid)), len(valid)

def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean."""
    values = np.asarray(values)
    valid = values[~np.isnan(values)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    
    rng = np.random.default_rng(seed)
    samples = rng.choice(valid, size=(n_boot, len(valid)), replace=True)
    means = np.mean(samples, axis=1)
    
    mean_val = np.mean(valid)
    lo = np.percentile(means, alpha / 2 * 100)
    hi = np.percentile(means, (1 - alpha / 2) * 100)
    
    return float(mean_val), float(lo), float(hi)

def calibrate_threshold(genuine: np.ndarray, impostor: np.ndarray, far: float = 0.01) -> dict[str, float]:
    """Calibrate similarity threshold for a given False Acceptance Rate (FAR)."""
    genuine = np.asarray(genuine)
    impostor = np.asarray(impostor)
    genuine = genuine[~np.isnan(genuine)]
    impostor = impostor[~np.isnan(impostor)]
    
    if len(genuine) == 0 or len(impostor) == 0:
        return {"tau_far": np.nan, "eer": np.nan, "tau_eer": np.nan, "tar_at_far": np.nan}
    
    tau_far = np.percentile(impostor, (1 - far) * 100)
    tar_at_far = np.mean(genuine >= tau_far)
    
    all_scores = np.concatenate([genuine, impostor])
    thresholds = np.sort(np.unique(all_scores))
    
    gen_sorted = np.sort(genuine)
    imp_sorted = np.sort(impostor)
    
    fnrs = np.searchsorted(gen_sorted, thresholds, side='left') / len(genuine)
    fprs = 1.0 - np.searchsorted(imp_sorted, thresholds, side='left') / len(impostor)
    
    diffs = np.abs(fprs - fnrs)
    min_idx = np.argmin(diffs)
    
    tau_eer = thresholds[min_idx]
    eer = (fprs[min_idx] + fnrs[min_idx]) / 2.0
            
    return {
        "tau_far": float(tau_far),
        "eer": float(eer),
        "tau_eer": float(tau_eer),
        "tar_at_far": float(tar_at_far)
    }
