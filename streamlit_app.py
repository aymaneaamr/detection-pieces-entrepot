"""
Application Streamlit pour la détection de pièces d'entrepôt
Déployable sur Streamlit Cloud
"""

import streamlit as st
import cv2
import numpy as np
import torch
import tempfile
import os
from PIL import Image
import pandas as pd
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Détection Pièces Entrepôt",
    page_icon="🏭",
    layout="wide"
)

# Titre
st.title("🏭 Système de Détection de Pièces d'Entrepôt")
st.markdown("---")

# Sidebar pour la configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Choix de la source
    source = st.radio(
        "Source d'image:",
        ["📷 Caméra", "📁 Upload image", "🎥 Vidéo"]
    )
    
    # Seuil de confiance
    confidence = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.markdown("---")
    st.header("📊 Base de données")
    
    # Informations sur les pièces
    piece_info = {
        "boulon": {"prix": "0.50€", "stock": 150, "emplacement": "A-12"},
        "vis": {"prix": "0.30€", "stock": 300, "emplacement": "B-03"},
        "ecrou": {"prix": "0.20€", "stock": 200, "emplacement": "C-07"},
        "rondelle": {"prix": "0.15€", "stock": 500, "emplacement": "A-05"},
        "clou": {"prix": "0.10€", "stock": 1000, "emplacement": "D-01"}
    }
    
    df_info = pd.DataFrame(piece_info).T
    st.dataframe(df_info)

# Chargement du modèle
@st.cache_resource
def load_model():
    """Charger le modèle YOLOv5"""
    try:
        # Essayer de charger le modèle personnalisé
        if os.path.exists("best.pt"):
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
        else:
            # Utiliser le modèle pré-entraîné
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

# Fonction de détection
def detect_objects(model, image, conf_threshold):
    """Détecter les objets dans l'image"""
    results = model(image)
    results = results.pandas().xyxy[0]
    results = results[results['confidence'] >= conf_threshold]
    return results

# Fonction pour dessiner les boîtes
def draw_boxes(image, detections):
    """Dessiner les boîtes de détection sur l'image"""
    img = image.copy()
    for _, det in detections.iterrows():
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        label = f"{det['name']} {det['confidence']:.2f}"
        
        # Dessiner rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Dessiner label
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img

# Interface principale
col1, col2 = st.columns(2)

with col1:
    st.header("📷 Image Source")
    
    if source == "📷 Caméra":
        # Capture caméra
        img_file = st.camera_input("Prendre une photo")
        if img_file is not None:
            bytes_data = img_file.getvalue()
            image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(image, caption="Image capturée", use_column_width=True)
            
    elif source == "📁 Upload image":
        # Upload d'image
        img_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
        if img_file is not None:
            image = Image.open(img_file)
            image = np.array(image)
            st.image(image, caption="Image uploadée", use_column_width=True)
            
    else:  # Vidéo
        video_file = st.file_uploader("Choisir une vidéo", type=['mp4', 'avi'])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.video(tfile.name)

with col2:
    st.header("🎯 Résultats Détection")
    
    if st.button("🚀 Lancer la détection"):
        with st.spinner("Détection en cours..."):
            # Charger modèle
            model = load_model()
            
            if model is not None and 'image' in locals():
                # Faire détection
                results = detect_objects(model, image, confidence)
                
                # Afficher résultats
                if len(results) > 0:
                    st.success(f"✅ {len(results)} pièces détectées!")
                    
                    # Dessiner boîtes
                    img_with_boxes = draw_boxes(image, results)
                    st.image(img_with_boxes, caption="Résultat détection", use_column_width=True)
                    
                    # Afficher tableau détails
                    st.subheader("📋 Détails des détections")
                    
                    # Ajouter infos supplémentaires
                    results['prix'] = results['name'].map(lambda x: piece_info.get(x, {}).get('prix', 'N/A'))
                    results['emplacement'] = results['name'].map(lambda x: piece_info.get(x, {}).get('emplacement', 'N/A'))
                    
                    st.dataframe(results[['name', 'confidence', 'prix', 'emplacement']])
                    
                    # Log des détections
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"Dernière détection: {timestamp}")
                    
                else:
                    st.warning("⚠️ Aucune pièce détectée")
            else:
                st.error("❌ Veuillez d'abord capturer/uploader une image")

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏭 Système de Détection de Pièces d'Entrepôt - YOLOv5 + Streamlit</p>
    <p>🔗 <a href='https://github.com/aymaneaamr/detection-pieces-entrepot'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
