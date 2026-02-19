"""
Script de détection en temps réel avec YOLOv5
"""

import torch
import cv2
import sqlite3
from datetime import datetime

def main():
    print("🔄 Chargement du modèle YOLOv5...")
    
    # Charger le modèle (à remplacer par ton modèle entraîné)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Base de données pour l'inventaire
    conn = sqlite3.connect('inventaire.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  piece TEXT, date TEXT, confiance REAL)''')
    
    # Démarrer caméra
    cap = cv2.VideoCapture(0)
    print("✅ Caméra démarrée. 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Détection
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        # Afficher les résultats
        for _, detection in detections.iterrows():
            piece = detection['name']
            conf = detection['confidence']
            
            if conf > 0.5:
                print(f"🔍 {piece} détecté (confiance: {conf:.2f})")
                
                # Sauvegarder dans la BD
                c.execute("INSERT INTO detections (piece, date, confiance) VALUES (?, ?, ?)",
                         (piece, datetime.now(), conf))
                conn.commit()
        
        # Afficher l'image
        cv2.imshow('Detection Entrepot', results.render()[0])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Statistiques
    print("\n📊 Statistiques:")
    for row in c.execute("SELECT piece, COUNT(*) FROM detections GROUP BY piece"):
        print(f"  {row[0]}: {row[1]} détections")
    
    conn.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
