import os

import numpy as np


def read_rgb(path: str) -> np.ndarray:
    """Read an image from path as RGB uint8."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_png(path: str, img: np.ndarray):
    """Write an RGB uint8 image to path as PNG."""
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(path, img_bgr)
    if not success:
        raise OSError(f"Failed to write image to {path}")

def to_tensor(img: np.ndarray):
    """Convert uint8 HWC image to float32 (1, 3, H, W) tensor in [0, 1]."""
    import torch
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    return t / 255.0

def to_uint8(tensor) -> np.ndarray:
    """Convert tensor back to uint8 HWC numpy array."""
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    img = (t * 255.0).clamp(0, 255).round().permute(1, 2, 0).numpy()
    return img.astype(np.uint8)

class ResumableCSV:
    """CSV writer that can resume, keeping track of completed keys."""
    def __init__(self, path: str, key_cols: list[str]):
        self.path = path
        self.key_cols = key_cols
        self.fieldnames = None
        self._keys = set()
        self._load_existing()

    def _load_existing(self):
        import csv
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(self.path):
            # A killed run (e.g. Kaggle session timeout) can leave a half-written last line;
            # drop it so that row is redone instead of being marked as done.
            with open(self.path, 'rb+') as f:
                data = f.read()
                if data and not data.endswith(b"\n"):
                    f.truncate(data.rfind(b"\n") + 1)
            with open(self.path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                self.fieldnames = reader.fieldnames
                for row in reader:
                    key = tuple(str(row.get(k, "")) for k in self.key_cols)
                    self._keys.add(key)

    def done_keys(self) -> set:
        """Return the set of completed keys (as tuples of strings)."""
        return self._keys

    def append(self, row: dict):
        import csv
        write_header = False
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            write_header = True
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
                
        with open(self.path, 'a', newline='') as f:
            if self.fieldnames is None:
                self.fieldnames = list(row.keys())
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
        
        key = tuple(str(row.get(k, "")) for k in self.key_cols)
        self._keys.add(key)
