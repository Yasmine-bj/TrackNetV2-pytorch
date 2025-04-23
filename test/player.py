import cv2
import numpy as np
from ultralytics import YOLO


# Charger le modèle YOLOv5 (yolo11m)
model = YOLO("yolo11m.pt")

# Définir le chemin de la vidéo
video_path = "/home/jovyan/TrackNetV2-pytorch/input/enregistrement.mp4"

# Charger la vidéo avec OpenCV
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Récupérer les dimensions de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir la sortie vidéo (nom du fichier, codec, FPS, dimensions)
output_video_path = "/home/jovyan/TrackNetV2-pytorch/output/detected_video.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

# Définir le polygone du terrain avec les 18 coordonnées directement dans un tableau NumPy
terrain_polygon = np.array([
    [652, 65],
    [608, 104],
    [454, 249],
    [298, 439],
    [153, 663],
    [35, 959],
    [442, 1065],
    [531, 1079],
    [1375, 1079],
    [1817, 982],
    [1693, 681],
    [1556, 453],
    [1415, 271],
    [1268, 112],
    [1228, 72],
    [1062, 62],
    [856, 58]
], np.int32).reshape((-1, 1, 2))

# Exécuter YOLO avec un seuil de confiance de 0.3 et filtrer uniquement la classe "person" (ID 0)
results = model.predict(source=video_path, stream=True, conf=0.3, classes=[0])

# Traiter chaque frame retournée
for r in results:
    detections = r.boxes
    for detection in detections:
        # Extraire les coordonnées de la boîte de détection : [x1, y1, x2, y2]
        x1, y1, x2, y2 = detection.xyxy[0]
        
        # Vérifier que les coins bas (bas gauche et bas droit) de la boîte se trouvent dans le polygone du terrain
        if (cv2.pointPolygonTest(terrain_polygon, (int(x1), int(y2)), False) >= 0 and 
            cv2.pointPolygonTest(terrain_polygon, (int(x2), int(y2)), False) >= 0):
            
            # Extraire la confiance et dessiner la boîte avec le texte correspondant
            confidence = detection.conf.item()
            cv2.rectangle(r.orig_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(r.orig_img, f"Player (Conf: {confidence:.2f})", (int(x1), int(y1)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Dessiner le polygone du terrain en rouge
    cv2.polylines(r.orig_img, [terrain_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Ajouter la frame modifiée à la vidéo de sortie
    out.write(r.orig_img)

# Libérer les ressources
cap.release()
out.release()


print(f"Vidéo traitée enregistrée dans : {output_video_path}")
