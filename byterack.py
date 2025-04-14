import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Charger le modèle YOLO (ici "yolo11m.pt")
model = YOLO("yolo11m.pt")

# Instancier le tracker ByteTrack via Supervision
tracker = sv.ByteTrack()

# Instancier les annotateurs pour les boîtes, les labels et la heat map (trace)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
#heat_map_annotator = sv.HeatMapAnnotator()

# Définir le polygone du terrain avec vos 18 coordonnées, directement dans un tableau NumPy
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

# Chemin de la vidéo source et ouverture via OpenCV
video_path = "input/enregistrement.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# Récupérer les dimensions de la vidéo
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Définir la vidéo de sortie
output_video_path = "output/detected_tracked_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Appliquer YOLO sur la frame en ne détectant que la classe "person" (classe 0) avec un seuil de confiance de 0.3
    results = model(frame, conf=0.3, classes=[0])[0]
    # Convertir les résultats de YOLO en objet Detections de Supervision
    detections = sv.Detections.from_ultralytics(results)
    
    # Filtrer les détections en conservant celles dont les coins bas (bas gauche et bas droit) se trouvent dans le polygone du terrain
    valid_indices = []
    if detections.xyxy.size > 0:
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = box
            if (cv2.pointPolygonTest(terrain_polygon, (int(x1), int(y2)), False) >= 0 and
                cv2.pointPolygonTest(terrain_polygon, (int(x2), int(y2)), False) >= 0):
                valid_indices.append(i)
        detections = detections[valid_indices]
    
    # Mise à jour du tracker avec les détections filtrées
    detections = tracker.update_with_detections(detections)
    
    # Création des labels pour afficher l'ID de suivi
    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    
    # Annoter la frame avec les boîtes et labels
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # Ajouter la heat map (annotation de la trace) sur la frame annotée
    #annotated_frame = heat_map_annotator.annotate(annotated_frame, detections=detections)
    
    # Dessiner le polygone du terrain en rouge pour référence
    cv2.polylines(annotated_frame, [terrain_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Écrire la frame annotée dans la vidéo de sortie
    out.write(annotated_frame)
    frame_id += 1

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vidéo traitée enregistrée dans : {output_video_path}")
