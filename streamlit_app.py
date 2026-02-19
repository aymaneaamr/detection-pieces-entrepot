"""
Application Streamlit avec YOLOv5 pour la détection de pièces
Compatible avec Streamlit Cloud
"""

import streamlit as st
import cv2
import numpy as np
import torch
import pandas as pd
from datetime import datetime
import os
import tempfile
from pathlib import Path
import requests
import zipfile
import shutil

# Configuration de la page
st.set_page_config(
    page_title="Détection Pièces Entrepôt - YOLOv5",
    page_icon="🔧",
    layout="wide"
)

# Titre
st.title("🔧 Détection de Pièces d'Entrepôt - YOLOv5")
st.markdown("---")

# Fonction pour télécharger YOLOv5
@st.cache_resource
def setup_yolov5():
    """Télécharge et configure YOLOv5"""
    try:
        if not os.path.exists('yolov5'):
            with st.spinner("📦 Téléchargement de YOLOv5..."):
                # Télécharger YOLOv5
                os.system('git clone https://github.com/ultralytics/yolov5.git')
                
                # Installer les requirements YOLOv5
                os.system('pip install -r yolov5/requirements.txt')
        
        # Retourner le chemin
        return os.path.abspath('yolov5')
    
    except Exception as e:
        st.error(f"Erreur setup YOLOv5: {e}")
        return None

# Fonction pour charger le modèle
@st.cache_resource
def load_model(model_type='yolov5s'):
    """Charge le modèle YOLOv5"""
    try:
        # Configuration du chemin YOLOv5
        yolov5_path = setup_yolov5()
        if yolov5_path:
            # Ajouter YOLOv5 au path
            import sys
            if yolov5_path not in sys.path:
                sys.path.append(yolov5_path)
            
            # Charger le modèle via torch.hub avec chemin local
            model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True, trust_repo=True)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Erreur chargement modèle: {e}")
        return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Choix du modèle
    model_choice = st.selectbox(
        "Modèle YOLO",
        ["yolov5n (nano - rapide)", 
         "yolov5s (small - équilibré)",
         "yolov5m (medium - précis)"],
        index=1
    )
    
    model_map = {
        "yolov5n (nano - rapide)": "yolov5n",
        "yolov5s (small - équilibré)": "yolov5s",
        "yolov5m (medium - précis)": "yolov5m"
    }
    
    # Seuil de confiance
    confidence = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    
    # Chargement du modèle
    if st.button("🚀 Charger le modèle", type="primary"):
        with st.spinner("Chargement du modèle..."):
            model = load_model(model_map[model_choice])
            if model:
                st.session_state['model'] = model
                st.session_state['model_loaded'] = True
                st.success("✅ Modèle chargé!")
    
    # Base de données pièces
    st.markdown("---")
    st.header("📦 Base pièces")
    
    if 'inventory' not in st.session_state:
        st.session_state.inventory = {
            'boulon': {'prix': '0.50€', 'stock': 150, 'empl': 'A12'},
            'vis': {'prix': '0.30€', 'stock': 300, 'empl': 'B03'},
            'ecrou': {'prix': '0.20€', 'stock': 200, 'empl': 'C07'},
            'rondelle': {'prix': '0.15€', 'stock': 500, 'empl': 'A05'},
            'clou': {'prix': '0.10€', 'stock': 1000, 'empl': 'D01'}
        }
    
    # Afficher inventaire
    df_inv = pd.DataFrame(st.session_state.inventory).T
    st.dataframe(df_inv, use_container_width=True)

# Interface principale
tab1, tab2, tab3 = st.tabs(["📷 Caméra", "📁 Upload", "📊 Stats"])

with tab1:
    st.header("Capture Caméra")
    
    # Capture
    img_file = st.camera_input("Prendre une photo")
    
    if img_file:
        # Lire l'image
        bytes_data = img_file.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_rgb, caption="Image originale", use_column_width=True)
        
        with col2:
            if st.button("🔍 Détecter", key="detect_cam"):
                if 'model' in st.session_state:
                    with st.spinner("Analyse YOLO en cours..."):
                        try:
                            # Détection
                            results = st.session_state.model(image_rgb)
                            
                            # Filtrer par confiance
                            detections = results.pandas().xyxy[0]
                            detections = detections[detections['confidence'] >= confidence]
                            
                            if len(detections) > 0:
                                st.success(f"✅ {len(detections)} pièces trouvées!")
                                
                                # Afficher résultat
                                img_result = results.render()[0]
                                st.image(img_result, caption="Résultat YOLO", use_column_width=True)
                                
                                # Détails
                                st.subheader("📋 Détections")
                                display_df = detections[['name', 'confidence']].copy()
                                display_df['confiance'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
                                
                                # Ajouter infos inventaire
                                display_df['prix'] = display_df['name'].map(
                                    lambda x: st.session_state.inventory.get(x, {}).get('prix', 'N/A'))
                                display_df['stock'] = display_df['name'].map(
                                    lambda x: st.session_state.inventory.get(x, {}).get('stock', 'N/A'))
                                
                                st.dataframe(display_df[['name', 'confiance', 'prix', 'stock']])
                            else:
                                st.warning("⚠️ Aucune pièce détectée")
                        
                        except Exception as e:
                            st.error(f"Erreur détection: {e}")
                else:
                    st.warning("⚠️ Charge d'abord le modèle dans la sidebar")

with tab2:
    st.header("Upload Image")
    
    uploaded = st.file_uploader("Choisir image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded:
        # Lire image
        image = Image.open(uploaded)
        image_np = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image_np, caption="Image originale", use_column_width=True)
        
        with col2:
            if st.button("🔍 Détecter", key="detect_upload"):
                if 'model' in st.session_state:
                    with st.spinner("Analyse YOLO en cours..."):
                        try:
                            results = st.session_state.model(image_np)
                            detections = results.pandas().xyxy[0]
                            detections = detections[detections['confidence'] >= confidence]
                            
                            if len(detections) > 0:
                                st.success(f"✅ {len(detections)} pièces trouvées!")
                                img_result = results.render()[0]
                                st.image(img_result, caption="Résultat YOLO", use_column_width=True)
                                
                                # Stats
                                st.subheader("📊 Comptage")
                                counts = detections['name'].value_counts()
                                for name, count in counts.items():
                                    st.metric(name, count)
                            else:
                                st.warning("⚠️ Aucune pièce détectée")
                        except Exception as e:
                            st.error(f"Erreur: {e}")
                else:
                    st.warning("⚠️ Charge d'abord le modèle")

with tab3:
    st.header("📊 Statistiques")
    
    if 'model_loaded' in st.session_state:
        st.info(f"✅ Modèle chargé: {model_choice}")
    
    # Graphique inventaire
    st.subheader("Niveaux de stock")
    stock_data = pd.DataFrame({
        'Pièce': list(st.session_state.inventory.keys()),
        'Stock': [v['stock'] for v in st.session_state.inventory.values()]
    })
    st.bar_chart(stock_data.set_index('Pièce'))
    
    # Valeur totale
    total_value = sum([float(v['prix'].replace('€', '')) * v['stock'] 
                      for v in st.session_state.inventory.values()])
    st.metric("Valeur totale stock", f"{total_value:.2f}€")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔧 <strong>YOLOv5 - Détection Pièces Entrepôt</strong></p>
    <p>1. Charge le modèle dans la sidebar | 2. Prends une photo | 3. Détecte !</p>
</div>
""", unsafe_allow_html=True)
