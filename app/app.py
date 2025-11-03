import json, torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from src.models import make_model
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@st.cache_resource
def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    arch = ckpt.get('arch','vit_base_patch16_224')
    num_classes = ckpt.get('num_classes',1)
    model = make_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    return model, num_classes

st.title('Histopath Fine-Tune Demo')
ckpt_path = st.sidebar.text_input('Checkpoint path', 'checkpoints/best.pt')
labels_file = st.sidebar.text_input('Labels file', 'runs/labels.json')
size = st.sidebar.number_input('Image size', min_value=128, max_value=512, value=224, step=16)

tf = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

if st.sidebar.button('Load Model'):
    st.session_state['model'], st.session_state['num_classes'] = load_model(ckpt_path)
    with open(labels_file,'r') as f:
        st.session_state['classes'] = json.load(f)['classes']
    st.success('Model loaded.')
    st.write("Classes:", st.session_state.get("classes"))
    st.write("num_classes:", st.session_state.get("num_classes"))


uploaded = st.file_uploader('Upload patch image', type=['png','jpg','jpeg','tif','tiff'])
if uploaded and 'model' in st.session_state:
    img = Image.open(uploaded).convert('RGB')
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        out = st.session_state['model'](x)
        if st.session_state['num_classes'] == 1:
            prob = torch.sigmoid(out.view(-1)).item()
            st.write(f'**Positive probability:** {prob:.4f}')
        else:
            prob = torch.softmax(out, dim=1).squeeze().tolist()
            import numpy as np
            pred = int(np.argmax(prob))
            st.write(f'**Predicted class:** {st.session_state["classes"][pred]}')
            st.write({st.session_state['classes'][i]: float(p) for i,p in enumerate(prob)})
    st.image(img, caption='Uploaded patch', use_column_width=True)
    
import glob, random
if st.button("Test with sample from data/test"):
    try:
        paths = glob.glob("data/test/*/*.*")
        if not paths:
            st.warning("No sample files found under data/test/")
        else:
            p = random.choice(paths)
            st.info(f"Testing with: {p}")
            from PIL import Image
            img = Image.open(p).convert("RGB")
            x = tf(img).unsqueeze(0)
            with torch.no_grad():
                out = st.session_state["model"](x)
                if st.session_state["num_classes"] == 1:
                    prob = torch.sigmoid(out.view(-1)).item()
                    st.write(f"**Positive probability:** {prob:.4f}")
                else:
                    prob = torch.softmax(out, dim=1).squeeze().tolist()
                    pred = max(range(len(prob)), key=lambda i: prob[i])
                    st.write(f"**Predicted class:** {st.session_state['classes'][pred]}")
                    st.write({st.session_state['classes'][i]: float(p) for i, p in enumerate(prob)})
            st.image(img, caption=os.path.basename(p), use_column_width=True)
    except Exception as e:
        st.error(f"Sample test failed: {e}")
