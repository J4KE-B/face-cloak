import math

import numpy as np


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

# ----------------------------------------------------------------------------- clustered statistics
def cluster_bootstrap_ci(
    values, clusters, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile CI of the pooled mean, resampling whole clusters (e.g. all rows of one source id).

    Rows of the same cluster are dependent (same source photo, several targets / transforms), so the
    i.i.d. row bootstrap understates uncertainty. NaN values are dropped first. The point estimate is the
    plain mean over valid rows; each replicate is ``sum(cluster sums) / sum(cluster counts)`` over the
    resampled clusters (rows keep equal weight, clusters with more valid rows weigh more).
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    keep = ~np.isnan(v)
    v, c = v[keep], c[keep]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    _, inv = np.unique(c, return_inverse=True)
    k = int(inv.max()) + 1
    sums = np.bincount(inv, weights=v, minlength=k)
    cnts = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    means = sums[idx].sum(1) / cnts[idx].sum(1)
    return (
        float(v.mean()),
        float(np.percentile(means, alpha / 2 * 100)),
        float(np.percentile(means, (1 - alpha / 2) * 100)),
    )


def wilson_interval(k: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n`` (well behaved at 0 and 1)."""
    if n <= 0:
        return np.nan, np.nan
    from statistics import NormalDist

    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # clamp so the interval always brackets p (rounding can put lo a hair above p = 0)
    return min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))


# ----------------------------------------------------------------------------- threshold calibration
def calibrate_far(genuine: np.ndarray, impostor: np.ndarray, far: float) -> float:
    """Smallest threshold tau (accept iff sim >= tau) with impostor FAR <= ``far``.

    The single FAR definition used everywhere (``scripts/calibrate.py``, :func:`calibrate_threshold`).
    ``genuine`` is unused but kept so call sites read ``calibrate_far(g, i, far)``.
    """
    imp = np.sort(np.asarray(impostor))[::-1]
    k = int(np.floor(far * len(imp)))  # number of impostors allowed above tau
    if k >= len(imp):
        return float(imp[-1])
    if k == 0:
        return float(np.nextafter(imp[0], np.inf))
    return float((imp[k - 1] + imp[k]) / 2.0)


def tar_at(genuine: np.ndarray, tau: float) -> float:
    return float(np.mean(np.asarray(genuine) >= tau))


def eer(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold."""
    g, i = np.sort(genuine), np.sort(impostor)
    taus = np.unique(np.concatenate([g, i]))
    frr = np.searchsorted(g, taus, side="left") / len(g)  # genuine rejected (sim < tau)
    far = 1.0 - np.searchsorted(i, taus, side="left") / len(i)  # impostor accepted
    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(taus[idx])


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold maximising accuracy (accept iff score >= tau)."""
    order = np.argsort(scores)
    s, y = scores[order], labels[order].astype(bool)
    # tau = s[k]: negatives below k correct, positives from k correct
    neg_below = np.concatenate([[0], np.cumsum(~y)])[:-1]
    pos_from = y.sum() - np.concatenate([[0], np.cumsum(y)])[:-1]
    return float(s[int(np.argmax(neg_below + pos_from))])


def kfold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> tuple[float, float]:
    accs = []
    for f in np.unique(folds):
        te = folds == f
        tau = best_threshold(scores[~te], labels[~te])
        accs.append(np.mean((scores[te] >= tau) == labels[te].astype(bool)))
    return float(np.mean(accs)), float(np.std(accs))


def calibrate_threshold(genuine: np.ndarray, impostor: np.ndarray, far: float = 0.01) -> dict[str, float]:
    """Thresholds for a target FAR plus EER; delegates to :func:`calibrate_far` / :func:`eer`."""
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    genuine = genuine[~np.isnan(genuine)]
    impostor = impostor[~np.isnan(impostor)]
    if len(genuine) == 0 or len(impostor) == 0:
        return {"tau_far": np.nan, "eer": np.nan, "tau_eer": np.nan, "tar_at_far": np.nan}
    tau_far = calibrate_far(genuine, impostor, far)
    e, tau_e = eer(genuine, impostor)
    return {"tau_far": tau_far, "eer": e, "tau_eer": tau_e, "tar_at_far": tar_at(genuine, tau_far)}
