import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from src.data import make_loaders
from src.models import make_model
from src.utils import get_device, EarlyStopper, save_label_map


def train_one_epoch(model, loader, crit, opt, device):
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        if logits.shape[1] == 1:
            logits = logits.view(-1)
            loss = crit(logits, y.float())
        else:
            loss = crit(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    total = 0.0
    ys, probs = [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if out.shape[1] == 1:
            out = out.view(-1)
            loss = crit(out, y.float())
            p = torch.sigmoid(out).cpu().numpy()
        else:
            loss = crit(out, y)
            p = torch.softmax(out, dim=1).cpu().numpy()
        total += loss.item() * x.size(0)
        ys.append(y.cpu().numpy()); probs.append(p)
    import numpy as np
    ys = np.concatenate(ys); probs = np.concatenate(probs)
    try:
        auc = roc_auc_score(ys, probs if probs.ndim == 1 else probs[:,0]) if probs.ndim == 1 or probs.shape[1]==1 else None
    except Exception:
        auc = None
    return total / len(loader.dataset), ys, probs, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data')
    ap.add_argument('--arch', type=str, default='vit_base_patch16_224')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--out_dir', type=str, default='checkpoints')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print('Device:', device)

    train_ld, val_ld, test_ld, classes = make_loaders(args.data_dir, batch_size=args.batch_size, size=args.img_size)
    num_classes = len(classes)
    binary = num_classes == 2

    model = make_model(args.arch, num_classes=1 if binary else num_classes, pretrained=True).to(device)
    crit = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    best_score = -1e9
    best_path = os.path.join(args.out_dir, 'best.pt')

    for epoch in range(1, args.epochs+1):
        tl = train_one_epoch(model, train_ld, crit, opt, device)
        vl, ys, probs, auc = validate(model, val_ld, crit, device)
        score = (auc if auc is not None else -vl)
        print(f'[Epoch {epoch}] train={tl:.4f} val={vl:.4f} score={score:.4f}')
        if score > best_score:
            best_score = score
            torch.save({'state_dict': model.state_dict(), 'arch': args.arch, 'num_classes': (1 if binary else num_classes)}, best_path)

    os.makedirs("runs", exist_ok=True) 
    save_label_map(classes, os.path.join('runs','labels.json'))
    print('Saved best to', best_path)

if __name__ == '__main__':
    main()
