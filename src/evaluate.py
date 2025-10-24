import os, argparse, json, numpy as np, torch, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc
from src.data import make_loaders
from src.models import make_model
from src.utils import get_device

def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, ylabel='True', xlabel='Pred')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center')
    plt.tight_layout(); plt.savefig(out_path); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, default='data')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--out_dir', type=str, default='runs')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    _, _, test_ld, classes = make_loaders(args.data_dir, batch_size=64, size=args.img_size)
    ckpt = torch.load(args.ckpt, map_location=device)
    arch = ckpt.get('arch','vit_base_patch16_224')
    num_classes = ckpt.get('num_classes', len(classes))

    from src.models import make_model
    model = make_model(arch, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    ys, probs = [], []
    with torch.no_grad():
        for x, y, _ in test_ld:
            x = x.to(device)
            out = model(x)
            if num_classes == 1:
                p = torch.sigmoid(out.view(-1)).cpu().numpy()
            else:
                p = torch.softmax(out, dim=1).cpu().numpy()
            ys.append(y.numpy()); probs.append(p)
    ys = np.concatenate(ys); probs = np.concatenate(probs)
    y_pred = (probs >= 0.5).astype(int) if num_classes == 1 else probs.argmax(1)

    plot_confusion(ys, y_pred, classes, os.path.join(args.out_dir, 'confusion_matrix.png'))
    if num_classes == 1:
        fpr, tpr, _ = roc_curve(ys, probs if probs.ndim==1 else probs[:,0])
        roc_auc = sk_auc(fpr, tpr)
        with open(os.path.join(args.out_dir,'metrics.json'),'w') as f:
            json.dump({'roc_auc': float(roc_auc)}, f)

if __name__ == '__main__':
    main()
