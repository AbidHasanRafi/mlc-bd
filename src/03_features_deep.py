"""Frozen-backbone embeddings for every image (row order == outputs/inventory.csv).
Images are RGB-converted and resized (bilinear, no cropping) to 224x224 and normalised with ImageNet statistics.
  resnet50 : torchvision IMAGENET1K_V1, global-average-pooled 2048-d
  dinov2_s : facebook/dinov2-small, CLS token 384-d"""
import sys, numpy as np, pandas as pd, torch, torchvision
from PIL import Image
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
torch.set_num_threads(4); torch.manual_seed(SEED)
inv = pd.read_csv(OUT / "inventory.csv")
paths = [DATA / r for r in inv.relpath]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

from concurrent.futures import ThreadPoolExecutor
def load(p):
    im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)
pool = ThreadPoolExecutor(4)
def batches(bs=64):
    for a in range(0, len(paths), bs):
        yield torch.stack(list(pool.map(load, paths[a:a + bs])))
dl = list(range(0, len(paths), 64))

r50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2); r50.fc = torch.nn.Identity(); r50.eval()
from transformers import AutoModel
dino = AutoModel.from_pretrained("facebook/dinov2-small").eval()
f1, f2 = [], []
with torch.inference_mode():
    for k, x in enumerate(batches()):
        x = (x.float() / 255 - mean) / std
        f1.append(r50(x).numpy()); f2.append(dino(pixel_values=x).last_hidden_state[:, 0].numpy())
        if k % 20 == 0: print(k, len(dl), flush=True)
np.save(OUT / "feat_resnet50.npy", np.concatenate(f1).astype(np.float32))
np.save(OUT / "feat_dinov2s.npy", np.concatenate(f2).astype(np.float32))
print("done")
