# Assurez-vous d'avoir installé ultralytics avec la commande :
# pip install ultralytics

from ultralytics import YOLO

# Charger le modèle YOLOv5 (yolo11x)
model = YOLO("yolo11x.pt")

# Définir le chemin de la vidéo
video_path = "/home/jovyan/TrackNetV2-pytorch/input/enregistrement.mp4"

# Exécuter YOLO avec un seuil de confiance réduit (conf=0.25)
results = model.predict(source=video_path, stream=True, save=True, conf=0.3)

for r in results:
    print("Vidéo traitée enregistrée dans:", r.save_dir)


