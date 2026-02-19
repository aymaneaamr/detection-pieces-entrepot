"""
Application Streamlit pour la détection de pièces d'entrepôt
Version corrigée - Utilisation de YOLOv5 via torch.hub
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
import requests
from pathlib import Path

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
    
    # Choix du modèle
    model_choice = st.selectbox(
        "Modèle",
        ["yolov5s (rapide)", "yolov5m (précis)", "yolov5n (très rapide)"]
    )
    
    model_map = {
        "yolov5s (rapide)": "yolov5s",
        "yolov5m (précis)": "yolov5m",
        "yolov5n (très rapide)": "yolov5n"
    }
    
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

# Fonction pour charger le modèle YOLOv5
@st.cache_resource
def load_model(model_name="yolov5s"):
    """Charger le modèle YOLOv5 via torch.hub"""
    try:
        with st.spinner(f"Chargement du modèle {model_name}..."):
            # Charger depuis torch hub
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            return model
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

# Fonction de détection
def detect_objects(model, image, conf_threshold):
    """Détecter les objets dans l'image"""
    if model is None:
        return None
    
    # Conversion de l'image si nécessaire
    if isinstance(image, np.ndarray):
        # Convertir BGR en RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # Faire la détection
    results = model(image_rgb)
    
    # Convertir en DataFrame
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] >= conf_threshold]
    
    return detections, results

# Fonction pour dessiner les boîtes
def draw_boxes(image, results):
    """Dessiner les boîtes de détection sur l'image"""
    if results is None:
        return image
    
    # Récupérer l'image avec les boîtes
    img_with_boxes = results.render()[0]
    return img_with_boxes

# Interface principale
col1, col2 = st.columns(2)

with col1:
    st.header("📷 Image Source")
    
    image = None
    video_path = None
    
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
        video_file = st.file_uploader("Choisir une vidéo", type=['mp4', 'avi', 'mov'])
        if video_file is not None:
            # Sauvegarder temporairement la vidéo
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(video_file.read())
            video_path = tfile.name
            st.video(video_path)

with col2:
    st.header("🎯 Résultats Détection")
    
    if st.button("🚀 Lancer la détection", type="primary"):
        if image is not None or video_path is not None:
            # Charger le modèle
            model_name = model_map[model_choice]
            model = load_model(model_name)
            
            if model is not None:
                if image is not None:
                    # Détection sur image
                    with st.spinner("Analyse de l'image en cours..."):
                        detections, results = detect_objects(model, image, confidence)
                        
                        if detections is not None and len(detections) > 0:
                            st.success(f"✅ {len(detections)} pièce(s) détectée(s)!")
                            
                            # Afficher l'image avec boîtes
                            img_with_boxes = draw_boxes(image, results)
                            st.image(img_with_boxes, caption="Résultat détection", use_column_width=True)
                            
                            # Afficher le tableau des détections
                            st.subheader("📋 Détails des détections")
                            
                            # Ajouter les informations des pièces
                            display_df = detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].copy()
                            display_df['confiance (%)'] = (display_df['confidence'] * 100).round(1)
                            
                            # Ajouter les infos de la base
                            display_df['prix'] = display_df['name'].map(lambda x: piece_info.get(x, {}).get('prix', 'N/A'))
                            display_df['emplacement'] = display_df['name'].map(lambda x: piece_info.get(x, {}).get('emplacement', 'N/A'))
                            
                            st.dataframe(display_df[['name', 'confiance (%)', 'prix', 'emplacement']])
                            
                            # Log de la détection
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.caption(f"🕐 Détection effectuée le: {timestamp}")
                            
                        else:
                            st.warning("⚠️ Aucune pièce détectée. Essayez d'ajuster le seuil de confiance.")
                            
                elif video_path is not None:
                    st.info("🎥 Détection sur vidéo - Fonctionnalité à venir...")
            else:
                st.error("❌ Impossible de charger le modèle")
        else:
            st.warning("⚠️ Veuillez d'abord capturer ou uploader une image")

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏭 <strong>Système de Détection de Pièces d'Entrepôt</strong> - YOLOv5 + Streamlit</p>
    <p>📸 Prenez une photo, uploadez une image ou une vidéo pour détecter automatiquement les pièces</p>
    <p>🔗 <a href='https://github.com/aymaneaamr/detection-pieces-entrepot' target='_blank'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)

# Instructions d'utilisation
with st.expander("📖 Comment utiliser cette application"):
    st.markdown("""
    ### Guide d'utilisation
    
    1. **Choisissez une source d'image** dans la barre latérale
    2. **Prenez une photo** avec votre caméra ou **uploader une image**
    3. **Ajustez le seuil de confiance** si nécessaire
    4. **Cliquez sur "Lancer la détection"**
    5. **Visualisez les résultats** avec les boîtes de détection
    
    ### Modèles disponibles
    - **yolov5n** : Très rapide, moins précis (nano)
    - **yolov5s** : Rapide, bon équilibre (small) - recommandé
    - **yolov5m** : Plus lent, plus précis (medium)
    
    ### Pièces détectables
    - Boulons, vis, écrous, rondelles, clous
    - Les informations (prix, stock, emplacement) sont affichées automatiquement
    """)
