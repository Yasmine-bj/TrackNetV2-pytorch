from ultralytics import YOLO

class YOLODetector:
    """Wrapper simplifié autour du modèle YOLO."""

    def __init__(self, model_path: str, conf: float = 0.001, classes=(0,)):
        self.model = YOLO(model_path)
        self.conf = conf
        self.classes = list(classes)

    def detect(self, frame):
        """Renvoie le résultat Ultralytics brut pour une frame."""
        return self.model(frame, conf=self.conf, classes=self.classes)[0]
