import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import os
import requests
import io

# --- Constants ---
MODEL_CACHE_DIR = "models/pretrained"
MODEL_DIR = "models/retrained"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(retrained_path, pretrained_path):
   import gdown
gdown.download(f"https://drive.google.com/drive/folders/1mVBIa3E1vNIsdIz7yZdW-lH3e8NJ4sXi?usp=sharing", retrained_path)
gdown.download(f"https://drive.google.com/drive/folders/1mVBIa3E1vNIsdIz7yZdW-lH3e8NJ4sXi?usp=sharing", pretrained_path)

MODEL_FILES = {
    "EfficientNet-B0": "efficientnet_b0_retrained.pth",
    "MobileNetV2": "mobilenet_v2_retrained.pth",
    "ResNet50": "resnet50_retrained.pth",
    "AlexNet": "alexnet_retrained.pth"
}

PRETRAINED_FILES = {
    "EfficientNet-B0": "efficientnet_b0.pth",
    "MobileNetV2": "mobilenet_v2.pth",
    "ResNet50": "resnet50.pth",
    "AlexNet": "alexnet.pth"
}

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy","background",
]


# --- Model Loading ---
def load_model(model_name):
    """Load model with proper error handling""" 
    try:
        # Initialize model
        if model_name == "EfficientNet-B0":
            model = torch.hub.load('pytorch/vision', 'efficientnet_b0', pretrained=False)
            model.classifier[1] = torch.nn.Linear(1280, len(CLASS_NAMES))
        elif model_name == "MobileNetV2":
            model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=False)
            model.classifier[1] = torch.nn.Linear(1280, len(CLASS_NAMES))
        elif model_name == "ResNet50":
            model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
            model.fc = torch.nn.Linear(2048, len(CLASS_NAMES))
        elif model_name == "AlexNet":
            model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=False)
            model.classifier[6] = torch.nn.Linear(4096, len(CLASS_NAMES))
        
        # Load weights
        pretrained_path = os.path.join(MODEL_CACHE_DIR, PRETRAINED_FILES[model_name])
        retrained_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
        
        if os.path.exists(pretrained_path):
            model.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
        if os.path.exists(retrained_path):
            model.load_state_dict(torch.load(retrained_path, map_location='cpu'))
        
        return model.eval()
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        return None

# --- Image Processing ---
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Streamlit UI ---
st.title("🌿 Plant Doctor Pro")
tab1, tab2, tab3 = st.tabs(["📷 Camera", "📂 Upload", "🌐 URL"])

# Camera Tab
with tab1:
    picture = st.camera_input("Take a leaf photo")
    if picture:
        img = Image.open(picture)
        if st.button("🔄 Flip Camera Image"):
            img = ImageOps.mirror(img)
        st.image(img, use_container_width=True)

# Upload Tab
with tab2:
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

# URL Tab
with tab3:
    url = st.text_input("Paste image URL")
    if url:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(io.BytesIO(response.content))
            st.image(img, use_container_width=True)
        except:
            st.error("Invalid URL or couldn't load image")

# Classification
if 'img' in locals():
    model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
    
    if st.button("🔍 Diagnose"):
        with st.spinner("Analyzing..."):
            model = load_model(model_name)
            if model:
                try:
                    img_tensor = TRANSFORM(img.convert("RGB")).unsqueeze(0)
                    with torch.no_grad():
                        output = model(img_tensor)
                        prediction = CLASS_NAMES[torch.argmax(output).item()]
                    st.success(f"**Diagnosis:** {prediction}")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
