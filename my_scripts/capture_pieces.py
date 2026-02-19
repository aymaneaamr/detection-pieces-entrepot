"""
Script pour capturer des photos de pièces avec la webcam
À utiliser sur ton PC (pas sur GitHub)
"""

import cv2
import os
from datetime import datetime

# Demander le nom de la pièce
piece = input("Nom de la pièce à photographier (ex: boulon) : ")

# Créer le dossier
dossier = f"dataset/{piece}"
os.makedirs(dossier, exist_ok=True)

# Démarrer la caméra
cap = cv2.VideoCapture(0)
compteur = 0

print(f"📸 Prêt à photographier des {piece}")
print("Appuie sur ESPACE pour prendre une photo")
print("Appuie sur 'q' pour quitter")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Afficher le compteur
    cv2.putText(frame, f"Photos: {compteur}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Capture - Appuie sur ESPACE', frame)
    
    key = cv2.waitKey(1)
    if key % 256 == 32:  # ESPACE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{dossier}/{piece}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        compteur += 1
        print(f"✅ Photo {compteur} sauvegardée")
    elif key % 256 == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n📊 Total: {compteur} photos de {piece}")
