import torch
import torch.nn.functional as F
from torch import nn
from transformers import SwinModel, AutoFeatureExtractor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import pandas as pd
import os
import io
import requests
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from transformers import SwinModel, AutoImageProcessor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# ----------------- Config -----------------
EXCEL_PATH = r"C:\Users\sammr\Desktop\Thesis\VQADataset\VQA_RAD_Head_AUGMENTED_2000.xlsx"
IMAGES_DIR = "VQA_RAD_Images"
GLOVE_PATH = "glove.6B.300d.txt"
CACHE_DIR = "patch_cache"
MODEL_OUT = "closed_qnan_model.pt"

SWIN_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"
SWIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NODE_DIM = 300
Q_DIM = 300
HIDDEN_DIM = 256
OUT_DIM = 256

KNN_K = 5
BATCH_SIZE = 1
LR = 5e-5
EPOCHS = 5


# ----------------- Utilities -----------------
def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def download_image(url: str, dst: Path, timeout=15):
    if dst.exists():
        return dst
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        img.save(dst)
        return dst
    except Exception as e:
        raise RuntimeError(f"Could not download {url}: {e}")


# ----------------- GloVe Embeddings-----------------
def load_glove(glove_path: str, verbose=True):
    glove = {}
    p = Path(glove_path)
    if not p.exists():
        raise FileNotFoundError(f"GloVe not found: {glove_path}")
    if verbose:
        print(f"[GloVe] Loading from {glove_path} ...")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec
    if verbose:
        print(f"[GloVe] Loaded {len(glove)} words.")
    return glove


def q2vec_mean(question: str, glove: dict, dim=300):
    toks = question.lower().split()
    vecs = [glove[t] for t in toks if t in glove]
    if len(vecs) == 0:
        return torch.zeros(dim, dtype=torch.float32)
    v = np.mean(vecs, axis=0)
    return torch.from_numpy(v).float()


# ----------------- Swin Feature Extractor -----------------
class SwinPatchExtractor:
    def __init__(self, model_name=SWIN_MODEL_NAME, cache_dir=CACHE_DIR, device=SWIN_DEVICE):
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = SwinModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def cache_path(self, image_path: str) -> Path:
        name = Path(image_path).stem
        return self.cache_dir / f"{name}.npy"

    def extract(self, image_path: str, reextract=False) -> torch.FloatTensor:
        cache_file = self.cache_path(image_path)
        if cache_file.exists() and not reextract:
            arr = np.load(cache_file)
            return torch.from_numpy(arr).float()
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
            patches = out.last_hidden_state.squeeze(0).cpu().numpy()
        np.save(cache_file, patches)
        return torch.from_numpy(patches).float()


# ----------------- KNN Graph -----------------
def build_knn_edge_index(node_feats: np.ndarray, k: int = 5):
    if torch.is_tensor(node_feats):
        feats = node_feats.cpu().numpy()
    else:
        feats = np.asarray(node_feats)
    N = len(feats)
    if N == 0:
        return torch.empty((2, 0), dtype=torch.long)
    k_eff = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="cosine").fit(feats)
    _, idxs = nbrs.kneighbors(feats)
    edges = []
    for i in range(N):
        for j in idxs[i]:
            if int(j) == i:
                continue
            edges.append((i, int(j)))
            edges.append((int(j), i))
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_idx


# ----------------- Dataset -----------------
class BrainVQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, glove: dict, swin_extractor: SwinPatchExtractor):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.glove = glove
        self.swin = swin_extractor
        ensure_dirs(self.images_dir)

    def __len__(self):
        return len(self.df)

    def _resolve_image(self, img_field: str) -> str:
        if isinstance(img_field, str) and img_field.startswith("http"):
            name = Path(img_field).name
            dst = self.images_dir / name
            if not dst.exists():
                download_image(img_field, dst)
            return str(dst)
        else:
            p = self.images_dir / str(img_field)
            if not p.exists():
                if Path(img_field).exists():
                    return str(img_field)
                raise FileNotFoundError(f"Image file not found: {p}")
            return str(p)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_field = row["IMAGEID"]
        q_text = str(row["QUESTION"])
        a_text = str(row["ANSWER"]).strip().lower()

        img_path = self._resolve_image(img_field)
        patch_feats = self.swin.extract(img_path)
        edge_index = build_knn_edge_index(patch_feats.numpy(), k=KNN_K)
        q_embed = q2vec_mean(q_text, self.glove, dim=Q_DIM)
        y = 1 if a_text.startswith("y") else 0

        data = Data(x=patch_feats, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
        data.q_embed = q_embed
        return data


# ----------------- Model QGNN-----------------
class QGNN(nn.Module):
    def __init__(self, swin_feat_dim: int, node_dim: int = NODE_DIM, q_dim: int = Q_DIM, hidden_dim: int = HIDDEN_DIM, out_dim: int = OUT_DIM):
        super().__init__()
        self.node_proj = nn.Linear(swin_feat_dim, node_dim)
        self.q_proj = nn.Linear(q_dim, node_dim)
        self.gcn1 = GCNConv(node_dim * 2, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x_raw, edge_index, q_embed):
        x_node = self.node_proj(x_raw)
        q_p = self.q_proj(q_embed)
        q_expand = q_p.unsqueeze(0).expand(x_node.size(0), -1)
        x_q = torch.cat([x_node, q_expand], dim=1)
        h = F.relu(self.gcn1(x_q, edge_index))
        h = self.gcn2(h, edge_index)
        return h


class AnswerClassifier(nn.Module):
    def __init__(self, qgnn: QGNN, num_classes: int = 2):
        super().__init__()
        self.qgnn = qgnn
        self.classifier = nn.Linear(qgnn.out_dim, num_classes)

    def forward(self, data: Data):
        node_emb = self.qgnn(data.x, data.edge_index, data.q_embed.to(data.x.device))
        batch = torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device)
        g_emb = global_mean_pool(node_emb, batch)
        logits = self.classifier(g_emb)
        return logits


# ----------------- Training + Eval -----------------
def train_and_eval(excel_path=EXCEL_PATH, images_dir=IMAGES_DIR, glove_path=GLOVE_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE):
    device = SWIN_DEVICE
    print(f"[TRAIN] device: {device}")

    df = pd.read_excel(excel_path)
    required_cols = {"IMAGEID", "QUESTION", "ANSWER"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Excel must contain columns: {required_cols}")

    glove = load_glove(glove_path)
    swin = SwinPatchExtractor(model_name=SWIN_MODEL_NAME, cache_dir=CACHE_DIR, device=device)

    dataset = BrainVQADataset(df, images_dir, glove, swin)
    n = len(dataset)
    indices = list(range(n))
    split = int(0.8 * n)
    train_idx, val_idx = indices[:split], indices[split:]
    from torch.utils.data import Subset
    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    sample = dataset[0]
    swin_feat_dim = sample.x.shape[1]
    print(f"[DATA] example patches: {sample.x.shape}, swin_feat_dim={swin_feat_dim}")

    qgnn = QGNN(swin_feat_dim=swin_feat_dim)
    model = AnswerClassifier(qgnn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    class_counts = np.bincount(df["ANSWER"].apply(lambda a: 1 if str(a).lower().startswith("y") else 0))
    weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_acc_list, val_acc_list = [], []

    for epoch in range(epochs):
        model.train()
        losses, y_true, y_pred = [], [], []
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}")
        for data in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            logits = model(data)
            target = data.y.to(device)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            pred = int(torch.argmax(logits, dim=1).item())
            y_pred.append(pred)
            y_true.append(int(target.item()))
            pbar.set_postfix({"loss": f"{np.mean(losses):.4f}", "acc": f"{accuracy_score(y_true, y_pred):.4f}"})

        train_acc = accuracy_score(y_true, y_pred)
        train_acc_list.append(train_acc)
        print(f"Epoch {epoch+1} Train acc {train_acc:.4f}")

        # validation
        model.eval()
        y_t_v, y_p_v = [], []
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Validation"):
                data = data.to(device)
                logits = model(data)
                pred = int(torch.argmax(logits, dim=1).item())
                y_p_v.append(pred)
                y_t_v.append(int(data.y.item()))
        val_acc = accuracy_score(y_t_v, y_p_v) if len(y_t_v) > 0 else 0.0
        val_acc_list.append(val_acc)
        print("Validation Acc:", val_acc)
        print(classification_report(y_t_v, y_p_v, digits=4))

        # confusion matrix
        cm = confusion_matrix(y_t_v, y_p_v, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.show()

    # plot accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), train_acc_list, label="Train Accuracy", marker="o")
    plt.plot(range(1, epochs+1), val_acc_list, label="Validation Accuracy", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


# ----------------- Main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=str, default=EXCEL_PATH)
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--glove", type=str, default=GLOVE_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    ensure_dirs(IMAGES_DIR, CACHE_DIR)
    train_and_eval(excel_path=args.excel, images_dir=args.images_dir, glove_path=args.glove, epochs=args.epochs)
