
import torch, torch.nn.functional as F

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self._acts = None; self._grads = None
        def fwd_hook(m, i, o): self._acts = o.detach()
        def bwd_hook(m, gi, go): self._grads = go[0].detach()
        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    @torch.no_grad()
    def _norm(self, x): return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def __call__(self, x, class_idx=None):
        x.requires_grad_(True)
        logits = self.model(x)
        if logits.shape[1] == 1:
            score = logits.view(-1).sigmoid().mean()
        else:
            if class_idx is None: class_idx = logits.argmax(1)
            score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        weights = self._grads.mean(dim=(2,3), keepdim=True)  # GAP over H,W
        cam = (weights * self._acts).sum(1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self._norm(cam[0,0]).cpu().numpy()
