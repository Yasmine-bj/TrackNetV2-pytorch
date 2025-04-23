import cv2
import supervision as sv
from ultralytics import YOLO
from constants.config import MODEL_PATH, CONF_THRES, TERRAIN_POLYGON

class YOLODetector:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))
    def detect(self, frame):
        return self.model(frame, conf=CONF_THRES, classes=[0])[0]

class ByteTracker:
    def __init__(self):
        self._trk = sv.ByteTrack()
    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._trk.update_with_detections(detections)

class Annotators:
    def __init__(self):
        from supervision import Color, BoxAnnotator, LabelAnnotator
        col = Color(0,0,0)
        self.box   = BoxAnnotator(color=col, thickness=2)
        self.label = LabelAnnotator(color=col)
    def apply(self, frame, detections, labels):
        f = self.box.annotate(frame, detections)
        return self.label.annotate(f, detections, labels=labels)


def filter_on_terrain(detections: sv.Detections) -> sv.Detections:
    valid = []
    for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
        if (cv2.pointPolygonTest(TERRAIN_POLYGON, (int(x1), int(y2)), False) >= 0
           and cv2.pointPolygonTest(TERRAIN_POLYGON, (int(x2), int(y2)), False) >= 0):
            valid.append(i)
    return detections[valid]