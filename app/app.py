import os, sys, json
from PIL import Image
import numpy as np
import streamlit as st
import torch
from torchvision import transforms

# --- ensure local src/ is importable BEFORE importing project modules ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import make_model
from src.gradcam import SimpleGradCAM  # requires src/gradcam.py to exist

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


@st.cache_resource
def load_model(ckpt_path):
    """
    Loads model checkpoint and returns (model, num_classes, arch).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = ckpt.get("arch", "vit_base_patch16_224")
    num_classes = ckpt.get("num_classes", 1)
    model = make_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, num_classes, arch


def make_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def overlay_cam_on_image(img_pil: Image.Image, cam: np.ndarray, alpha: int = 120) -> Image.Image:
    """
    img_pil: original RGB PIL image (any size)
    cam: HxW in [0,1]
    alpha: transparency of heatmap overlay (0-255)
    """
    base = img_pil.resize((cam.shape[1], cam.shape[0])).convert("RGBA")
    heat = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heat_rgb = np.stack([heat, np.zeros_like(heat), np.zeros_like(heat)], axis=-1)  # red channel
    heat_img = Image.fromarray(heat_rgb).convert("RGBA")
    heat_img.putalpha(alpha)
    return Image.alpha_composite(base, heat_img)


def try_load_labels(labels_file: str):
    try:
        with open(labels_file, "r") as f:
            obj = json.load(f)
        return obj.get("classes", None)
    except Exception:
        return None


# --------------------- UI ---------------------
st.title("Histopath Fine-Tune Demo")

ckpt_path   = st.sidebar.text_input("Checkpoint path", "checkpoints/best.pt")
labels_file = st.sidebar.text_input("Labels file", "runs/labels.json")
size        = st.sidebar.number_input("Image size", min_value=128, max_value=512, value=224, step=16)

# Grad-CAM toggle
show_cam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)

# Load/refresh model
if st.sidebar.button("Load Model"):
    st.session_state["model"], st.session_state["num_classes"], st.session_state["arch"] = load_model(ckpt_path)
    st.session_state["classes"] = try_load_labels(labels_file)
    st.success("Model loaded.")
    st.write("Classes:", st.session_state.get("classes"))
    st.write("num_classes:", st.session_state.get("num_classes"))
    st.write("arch:", st.session_state.get("arch"))

# Common transform (rebuilt each run so 'size' updates)
tf = make_transform(size)

# ------------- Single patch upload -------------
uploaded = st.file_uploader("Upload patch image", type=["png","jpg","jpeg","tif","tiff"])
if uploaded and "model" in st.session_state:
    img = Image.open(uploaded).convert("RGB")
    x = tf(img).unsqueeze(0)

    with torch.no_grad():
        out = st.session_state["model"](x)
        if st.session_state["num_classes"] == 1:
            prob_pos = torch.sigmoid(out.view(-1)).item()
            st.write(f"**Probability (Tumor):** {prob_pos:.4f}")
        else:
            probs = torch.softmax(out, dim=1).squeeze().tolist()
            pred  = int(np.argmax(probs))
            classes = st.session_state.get("classes", [f"class_{i}" for i in range(len(probs))])
            st.write(f"**Predicted class:** {classes[pred]}")
            st.write({classes[i]: float(p) for i, p in enumerate(probs)})

    st.image(img, caption="Uploaded patch", use_column_width=True)

    # --------- Grad-CAM for uploaded patch ----------
    if show_cam:
        model = st.session_state["model"]
        arch  = st.session_state.get("arch", "resnet18")

        # pick a target layer for ResNet-like models
        target_layer = getattr(getattr(model, "layer4", None), "__getitem__", None)
        if target_layer is not None:
            try:
                # model.layer4[-1].conv2 exists on resnet18/34
                tl = model.layer4[-1].conv2
                cam_map = SimpleGradCAM(model, tl)(x)
                overlay = overlay_cam_on_image(img, cam_map)
                st.image(overlay, caption="Grad-CAM (red = higher contribution)", use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated for this architecture. ({e})")
        else:
            st.info("Grad-CAM demo currently supports ResNet-style backbones. (ViT support: attention rollout to be added.)")

# ------------- Quick test from data/test -------------
import glob, random
if st.button("Test with sample from data/test"):
    if "model" not in st.session_state:
        st.error("Load the model first from the sidebar.")
    else:
        try:
            paths = glob.glob("data/test/*/*.*")
            if not paths:
                st.warning("No sample files found under data/test/")
            else:
                p = random.choice(paths)
                st.info(f"Testing with: {p}")
                img2 = Image.open(p).convert("RGB")
                x2 = tf(img2).unsqueeze(0)

                with torch.no_grad():
                    out2 = st.session_state["model"](x2)
                    if st.session_state["num_classes"] == 1:
                        prob2 = torch.sigmoid(out2.view(-1)).item()
                        st.write(f"**Probability (Tumor):** {prob2:.4f}")
                    else:
                        probs2 = torch.softmax(out2, dim=1).squeeze().tolist()
                        pred2  = int(np.argmax(probs2))
                        classes = st.session_state.get("classes", [f"class_{i}" for i in range(len(probs2))])
                        st.write(f"**Predicted class:** {classes[pred2]}")
                        st.write({classes[i]: float(pv) for i, pv in enumerate(probs2)})

                st.image(img2, caption=os.path.basename(p), use_column_width=True)

                # Grad-CAM on sample
                if show_cam:
                    model = st.session_state["model"]
                    target_layer = getattr(getattr(model, "layer4", None), "__getitem__", None)
                    if target_layer is not None:
                        try:
                            tl = model.layer4[-1].conv2
                            cam_map2 = SimpleGradCAM(model, tl)(x2)
                            overlay2 = overlay_cam_on_image(img2, cam_map2)
                            st.image(overlay2, caption="Grad-CAM (sample) — red = higher contribution", use_column_width=True)
                        except Exception as e:
                            st.warning(f"Grad-CAM (sample) failed: {e}")
                    else:
                        st.info("Grad-CAM demo currently supports ResNet-style backbones. (ViT support: attention rollout to be added.)")
        except Exception as e:
            st.error(f"Sample test failed: {e}")

