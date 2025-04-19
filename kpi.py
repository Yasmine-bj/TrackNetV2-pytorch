import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pandas as pd
from supervision import Color 

# Charger le modèle YOLO (ici "yolo11m.pt")
model = YOLO("yolo11m.pt")

# Instancier le tracker ByteTrack via Supervision
tracker = sv.ByteTrack()

# Instancier les annotateurs pour les boîtes, les labels et la heat map (trace)
box_annotator = sv.BoxAnnotator(color=Color(0, 0, 0), thickness=2) 
label_annotator = sv.LabelAnnotator(color=Color(0, 0, 0))
ellipse_annotator = sv.EllipseAnnotator()
corner_annotator = sv.BoxCornerAnnotator()



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

# Définir les zones d'attaque et de défense avec les 3 (ou plus) points
zone_attaque_polygon = np.array([
    [608, 104],
    [454, 249],
    [298, 439],
    [153, 663],
    [912, 695],
    [1693, 681],
    [1556, 453],
    [1415, 271],
    [1268, 112],
    #[926, 249]
    [937, 95]

], np.int32).reshape((-1, 1, 2))

zone_defense_polygon1 = np.array([
    [153, 663],
    [35, 959],
    [442, 1065],
    [531, 1079],
    [1375, 1079],
    [1817, 982],
    [1693, 681],
    [912, 695]
], np.int32).reshape((-1, 1, 2))

zone_defense_polygon2 = np.array([
    [652, 65],
    [608, 104],
    [937, 95],
    [1268, 112],
    [1228, 72],
    [1062, 62],
    [856, 58]
], np.int32).reshape((-1, 1, 2))


# Initialisation de la structure de données pour le CSV
data = []

# Fonction pour enregistrer la position et les informations du joueur
def record_player_data(frame_num, player_id, x, y, is_attack, is_defense):
    data.append({
        'frame_num': frame_num,
        'player_id': player_id,
        'x': x,
        'y': y,
        'attack': int(is_attack),
        'defense': int(is_defense)
    })

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
output_video_path = "output/kpi.mp4"
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
    #annotated_frame = corner_annotator.annotate( scene=frame.copy(), detections=detections)
    #annotated_frame = ellipse_annotator.annotate(scene=frame.copy(),  detections=detections )

    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    

      
    # Vérifier la position de chaque joueur par rapport aux zones d'attaque et de défense
    for i in range(len(detections.xyxy)):  # Accédez à chaque détection dans les attributs
        x1, y1, x2, y2 = detections.xyxy[i]  # coordonnées de la boîte de détection
        # Calculer le point (x, y) basé sur les coordonnées de la boîte
        x = (x1 + x2) / 2  # Point x du centre bas de la boîte
        y = y2  # Point y du coin inférieur de la boîte

        # Vérification de la position du joueur avec le point (x, y)
        is_attack = (cv2.pointPolygonTest(zone_attaque_polygon, (int(x), int(y)), False) >= 0)
        is_defense = (cv2.pointPolygonTest(zone_defense_polygon1, (int(x), int(y)), False) >= 0 or
                      cv2.pointPolygonTest(zone_defense_polygon2, (int(x), int(y)), False) >= 0)
        # Enregistrer les données pour le joueur
        record_player_data(frame_id, detections.tracker_id[i], x, y, is_attack, is_defense)

        # Annoter la frame avec l'état du joueur (attaquant ou défenseur)
        if is_attack:
            cv2.putText(annotated_frame, "Attaquant", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif is_defense:
            cv2.putText(annotated_frame, "Defenseur", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    # Dessiner le polygone du terrain en rouge pour référence
    cv2.polylines(annotated_frame, [terrain_polygon], isClosed=True, color=(0, 0, 255), thickness=2)
    # Convertir l'image à un format avec un canal alpha (transparence)
    #annotated_frame_with_alpha = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2BGRA)

    # Couleur semi-transparente pour la zone d'attaque (vert)
    #color_attack = (0, 255, 0, 80)  # RGBA : couleur verte avec 80/255 d'opacité


    # Couleur semi-transparente pour la zone de défense (rouge)
    #color_defense = (0, 0, 255, 80)  # RGBA : couleur rouge avec 80/255 d'opacité

    # Remplir la zone d'attaque avec la couleur semi-transparente
    #cv2.fillPoly(annotated_frame_with_alpha, [zone_attaque_polygon], color_attack)

    # Remplir la zone de défense avec la couleur semi-transparente
    #cv2.fillPoly(annotated_frame_with_alpha, [zone_defense_polygon], color_defense)

    # Convertir l'image de nouveau en BGR (pour afficher ou sauver l'image sans alpha)
    #annotated_frame = cv2.cvtColor(annotated_frame_with_alpha, cv2.COLOR_BGRA2BGR)

    # Dessiner le polygone du terrain en rouge pour référence
    cv2.polylines(annotated_frame, [zone_attaque_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(annotated_frame, [zone_defense_polygon1], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.polylines(annotated_frame, [zone_defense_polygon2], isClosed=True, color=(0, 0, 255), thickness=2)

    # Écrire la frame annotée dans la vidéo de sortie
    out.write(annotated_frame)
    frame_id += 1

cap.release()
out.release()


# Enregistrer les données dans un fichier CSV
df = pd.DataFrame(data)
df.to_csv('player_positions.csv', index=False)

# Calcul des pourcentages d'attaquant et de défenseur pour chaque joueur
total_frames = len(df['frame_num'].unique())
players = df['player_id'].unique()
for player_id in players:
    player_data = df[df['player_id'] == player_id]
    attack_time = player_data['attack'].sum() / total_frames * 100
    defense_time = player_data['defense'].sum() / total_frames * 100
    print(f"Player {player_id}: Attack time = {attack_time:.2f}% | Defense time = {defense_time:.2f}%")


print(f"Vidéo traitée enregistrée dans : {output_video_path}")
