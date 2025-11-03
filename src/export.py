import argparse, torch
from src.models import make_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out', type=str, default='checkpoints/best_scripted.pt')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    arch = ckpt.get('arch','vit_base_patch16_224')
    num_classes = ckpt.get('num_classes', 1)
    model = make_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['state_dict']); model.eval()

    example = torch.randn(1,3,224,224)
    scripted = torch.jit.trace(model, example)
    scripted.save(args.out)
    print('Saved TorchScript to', args.out)

if __name__ == '__main__':
    main()
