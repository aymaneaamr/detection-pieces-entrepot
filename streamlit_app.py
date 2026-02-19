"""
Application Streamlit pour la détection de pièces d'entrepôt
Version simplifiée sans torch.hub
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import tempfile

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
    
    # Seuil de confiance
    confidence = st.slider(
        "Seuil de confiance (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    ) / 100
    
    st.markdown("---")
    st.header("📊 Base de données pièces")
    
    # Informations sur les pièces
    piece_info = {
        "boulon": {"prix": "0.50€", "stock": 150, "emplacement": "A-12"},
        "vis": {"prix": "0.30€", "stock": 300, "emplacement": "B-03"},
        "ecrou": {"prix": "0.20€", "stock": 200, "emplacement": "C-07"},
        "rondelle": {"prix": "0.15€", "stock": 500, "emplacement": "A-05"},
        "clou": {"prix": "0.10€", "stock": 1000, "emplacement": "D-01"}
    }
    
    # Ajouter une nouvelle pièce
    with st.expander("➕ Ajouter une pièce"):
        new_piece = st.text_input("Nom")
        new_price = st.text_input("Prix")
        new_stock = st.number_input("Stock", min_value=0)
        new_location = st.text_input("Emplacement")
        if st.button("Ajouter"):
            if new_piece and new_price and new_location:
                piece_info[new_piece] = {
                    "prix": new_price,
                    "stock": new_stock,
                    "emplacement": new_location
                }
                st.success(f"✅ Pièce {new_piece} ajoutée!")
    
    # Afficher le tableau
    df_info = pd.DataFrame(piece_info).T
    st.dataframe(df_info, use_container_width=True)

# Fonction de détection simulée (pour test)
def simulate_detection(image, conf_threshold):
    """Simule une détection (version sans YOLO)"""
    height, width = image.shape[:2] if len(image.shape) == 3 else (image.shape[0], image.shape[1])
    
    # Simuler des détections aléatoires pour la démo
    import random
    pieces = list(piece_info.keys())
    num_detections = random.randint(0, 3)
    
    detections = []
    for i in range(num_detections):
        piece = random.choice(pieces)
        conf = random.uniform(conf_threshold, 1.0)
        
        # Boîte aléatoire
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        x2 = x1 + random.randint(50, 150)
        y2 = y1 + random.randint(50, 150)
        
        detections.append({
            'name': piece,
            'confidence': conf,
            'xmin': x1,
            'ymin': y1,
            'xmax': x2,
            'ymax': y2
        })
    
    return pd.DataFrame(detections)

# Fonction pour dessiner les boîtes
def draw_boxes(image, detections):
    """Dessiner les boîtes de détection sur l'image"""
    img = image.copy()
    
    for _, det in detections.iterrows():
        x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
        conf = det['confidence']
        name = det['name']
        
        # Couleur différente par type de pièce
        colors = {
            'boulon': (0, 255, 0),    # Vert
            'vis': (255, 0, 0),        # Bleu
            'ecrou': (0, 0, 255),      # Rouge
            'rondelle': (255, 255, 0), # Jaune
            'clou': (255, 0, 255)      # Magenta
        }
        color = colors.get(name, (0, 255, 0))
        
        # Dessiner rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Préparer le label
        label = f"{name} {conf:.0%}"
        
        # Dessiner le fond du texte
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        
        # Dessiner le texte
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 2)
    
    return img

# Interface principale
tab1, tab2, tab3 = st.tabs(["📷 Caméra", "📁 Upload Image", "📊 Statistiques"])

with tab1:
    st.header("Capture caméra")
    
    # Capture caméra
    img_file = st.camera_input("Prendre une photo")
    
    if img_file is not None:
        # Lire l'image
        bytes_data = img_file.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image originale")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Résultat détection")
            if st.button("🔍 Détecter les pièces", key="detect_cam"):
                with st.spinner("Analyse en cours..."):
                    # Simulation de détection
                    detections = simulate_detection(image, confidence)
                    
                    if len(detections) > 0:
                        st.success(f"✅ {len(detections)} pièce(s) détectée(s)!")
                        
                        # Dessiner les boîtes
                        img_result = draw_boxes(image, detections)
                        st.image(img_result, use_column_width=True)
                        
                        # Tableau des détections
                        st.subheader("📋 Détails")
                        display_df = detections[['name', 'confidence']].copy()
                        display_df['confiance'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
                        
                        # Ajouter infos
                        display_df['prix'] = display_df['name'].map(lambda x: piece_info.get(x, {}).get('prix', 'N/A'))
                        display_df['emplacement'] = display_df['name'].map(lambda x: piece_info.get(x, {}).get('emplacement', 'N/A'))
                        
                        st.dataframe(display_df[['name', 'confiance', 'prix', 'emplacement']], 
                                   use_container_width=True)
                        
                        # Log
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"🕐 {timestamp}")
                    else:
                        st.warning("⚠️ Aucune pièce détectée")

with tab2:
    st.header("Upload image")
    
    # Upload d'image
    uploaded_file = st.file_uploader("Choisir une image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Lire l'image
        image = Image.open(uploaded_file)
        image = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Image originale")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Résultat détection")
            if st.button("🔍 Détecter les pièces", key="detect_upload"):
                with st.spinner("Analyse en cours..."):
                    # Simulation de détection
                    detections = simulate_detection(image, confidence)
                    
                    if len(detections) > 0:
                        st.success(f"✅ {len(detections)} pièce(s) détectée(s)!")
                        
                        # Dessiner les boîtes
                        img_result = draw_boxes(image, detections)
                        st.image(img_result, use_column_width=True)
                        
                        # Tableau des détections
                        st.subheader("📋 Détails")
                        display_df = detections[['name', 'confidence']].copy()
                        display_df['confiance'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
                        
                        # Ajouter infos
                        display_df['prix'] = display_df['name'].map(lambda x: piece_info.get(x, {}).get('prix', 'N/A'))
                        display_df['emplacement'] = display_df['name'].map(lambda x: piece_info.get(x, {}).get('emplacement', 'N/A'))
                        
                        st.dataframe(display_df[['name', 'confiance', 'prix', 'emplacement']], 
                                   use_container_width=True)
                        
                        # Graphique
                        st.subheader("📊 Répartition")
                        chart_data = detections['name'].value_counts()
                        st.bar_chart(chart_data)
                        
                        # Log
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"🕐 {timestamp}")
                    else:
                        st.warning("⚠️ Aucune pièce détectée")

with tab3:
    st.header("📊 Statistiques globales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre de pièces", len(piece_info))
    
    with col2:
        total_stock = sum([p['stock'] for p in piece_info.values()])
        st.metric("Stock total", total_stock)
    
    with col3:
        avg_price = sum([float(p['prix'].replace('€', '')) for p in piece_info.values()]) / len(piece_info)
        st.metric("Prix moyen", f"{avg_price:.2f}€")
    
    # Tableau complet
    st.subheader("Inventaire complet")
    st.dataframe(df_info, use_container_width=True)
    
    # Graphique des stocks
    st.subheader("Niveaux de stock")
    stock_data = pd.DataFrame({
        'Pièce': list(piece_info.keys()),
        'Stock': [p['stock'] for p in piece_info.values()]
    })
    st.bar_chart(stock_data.set_index('Pièce'))

# Pied de page
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🏭 <strong>Système de Détection de Pièces d'Entrepôt</strong> - Version Démo</p>
    <p>📸 Prenez une photo ou uploadez une image pour simuler la détection</p>
    <p>🔧 <em>Version sans YOLO pour compatibilité Streamlit Cloud</em></p>
    <p>🔗 <a href='https://github.com/aymaneaamr/detection-pieces-entrepot' target='_blank'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)

# Instructions
with st.expander("ℹ️ Comment ça marche"):
    st.markdown("""
    ### Version Démo
    Cette version utilise une **simulation de détection** pour démontrer l'interface.
    
    ### Fonctionnalités
    - ✅ Interface complète
    - ✅ Base de données des pièces
    - ✅ Simulation de détection
    - ✅ Gestion d'inventaire
    - ✅ Ajout de nouvelles pièces
    
    ### Pour la version réelle avec YOLO
    La version avec véritable détection YOLOv5 nécessite :
    - Installation locale
    - GPU recommandé
    - Plus de ressources mémoire
    
    ### Prochaines étapes
    1. Ajoute tes vraies photos dans `dataset/`
    2. Entraîne le modèle YOLOv5
    3. Remplace la simulation par le vrai modèle
    """)
