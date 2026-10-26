"""Image branch: signal-derived STFT spectrogram / CWT scalogram -> frozen CNN embedding.

These images are computed from the vibration window itself (not photographs).
"""
import hashlib

import numpy as np
import pywt
import torch
import torch.nn.functional as F
import torchvision
from matplotlib import colormaps
from scipy.signal import stft

from . import PROC

SIZE = 224


def _to_rgb(img: np.ndarray) -> np.ndarray:
    """log-magnitude [H, W] -> min-max -> viridis RGB [3, 224, 224] float32."""
    img = np.log1p(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    t = torch.from_numpy(img.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()
    return colormaps["viridis"](t)[..., :3].transpose(2, 0, 1).astype(np.float32)


def stft_image(x: np.ndarray, fs: float) -> np.ndarray:
    _, _, z = stft(x, fs=fs, nperseg=128, noverlap=96)
    return _to_rgb(np.abs(z))


def cwt_image(x: np.ndarray, fs: float) -> np.ndarray:
    coef, _ = pywt.cwt(x, np.arange(1, 65), "morl", sampling_period=1 / fs)
    return _to_rgb(np.abs(coef))


def heatmap_image(m: np.ndarray, fs: float = 0) -> np.ndarray:
    """Generic [time, sensor] window rendered as a heatmap (used for C-MAPSS)."""
    return _to_rgb(m - m.min())


def make_images(sig: np.ndarray, fs: float, kind: str) -> np.ndarray:
    fn = {"stft": stft_image, "cwt": cwt_image, "heatmap": heatmap_image}[kind]
    return np.stack([fn(w[0] if kind != 'heatmap' else w, fs) for w in sig])


def _cnn(name: str = "resnet18"):
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        m.fc = torch.nn.Identity()
    else:
        m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        m.classifier = torch.nn.Identity()
    return m.eval()


def embed(sig: np.ndarray, fs: float, kind: str, tag: str, cnn: str = "resnet18", batch: int = 64) -> np.ndarray:
    """Frozen-CNN embeddings [N, 512], cached in data/processed/ under a content hash."""
    key = hashlib.md5(f"{tag}|{kind}|{cnn}|{fs}|{SIZE}".encode() + sig[:: max(1, len(sig) // 50)].tobytes()).hexdigest()[:10]
    path = PROC / f"emb_{tag}_{kind}_{cnn}_{key}.npy"
    if path.exists():
        return np.load(path)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = _cnn(cnn).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(sig), batch):
            imgs = torch.from_numpy(make_images(sig[i:i + batch], fs, kind)).to(dev)
            out.append(net((imgs - mean) / std).cpu().numpy())
            if (i // batch) % 10 == 0:
                print(f"  embed {tag}: {i}/{len(sig)}", flush=True)
    emb = np.concatenate(out)
    PROC.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
