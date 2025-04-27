import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import deque

from supervision.detection.overlap_filter import box_non_max_merge
from constants.config import (
    MODEL_PATH,
    CONF_THRES,
    TERRAIN_POLYGON,
    # ajoutez ceci si vous voulez paramétrer le seuil d’IoU pour le merge
    # OVERLAP_IOU_THRESH
)

class YOLODetector:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))
        
    def detect(self, frame):
        # ne renvoie que la classe 0 (person)
        return self.model(frame, conf=CONF_THRES, classes=[0])[0]

class ByteTracker:
    def __init__(self):
        # vos paramètres de tracking déjà ajustés
        self._trk = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=100,
            minimum_matching_threshold=0.75
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        # 1) Construction du tableau preds pour overlap_filter
        if detections.xyxy.size:
            # xyxy: (N,4), confidence: (N,), class_id: (N,)
            preds = np.concatenate([
                detections.xyxy,
                detections.confidence.reshape(-1,1),
                detections.class_id.reshape(-1,1)
            ], axis=1)  # shape (N,6)

            # 2) Application du merge non-maximal (iou_threshold fixé à 0.5)
            groups = box_non_max_merge(preds, iou_threshold=0.3)

            # 3) Reconstruction d’un nouveau sv.Detections fusionné
            merged = []
            for group in groups:
                idxs = list(group)
                block = preds[idxs]
                # moyennisation des boîtes
                x1, y1, x2, y2 = block[:,:4].mean(axis=0)
                # on garde la confiance maximale du groupe
                confs = block[:,4]
                best = int(confs.argmax())
                score = confs.max()
                # et la classe associée
                cls   = int(block[best,5])
                merged.append([x1, y1, x2, y2, score, cls])

            merged = np.array(merged)
            detections = sv.Detections(
                xyxy       = merged[:,:4],
                confidence = merged[:,4],
                class_id   = merged[:,5].astype(int)
            )

        # 4) On peut maintenant lancer le tracker sur ces détections fusionnées
        return self._trk.update_with_detections(detections)

# Gestion des IDs (pool fixe de 4 joueurs) avec proximité spatiale
class TrackIDManager:
    def __init__(self, pool_size=4):
        self.pool_ids = list(range(1, pool_size + 1))
        # mapping internal tracker ID -> assigned ID
        self.internal_to_assigned = {}
        # last known positions of assigned IDs: assigned ID -> (x, y)
        self.last_positions = {}
        # queue des IDs libres (sans ordre fixe)
        self.freed_ids = deque(self.pool_ids)
        # internals actifs de la frame précédente
        self.prev_internals = set()

    def update(self, current_internals, centroids):
        """
        current_internals: list of tracker internal IDs
        centroids: dict internal ID -> (x, y) for this frame
        """
        new_mapping = {}
        # 1) libérer les IDs pour les internes disparus
        disappeared = self.prev_internals - set(current_internals)
        for old in disappeared:
            assigned = self.internal_to_assigned.pop(old)
            # stocke position pour réassignation
            self.last_positions[assigned] = self.last_positions.get(assigned)
            self.freed_ids.append(assigned)

        # 2) conserver mapping pour internals toujours présents
        for tid, aid in self.internal_to_assigned.items():
            new_mapping[tid] = aid

        # 3) assigner IDs aux nouveaux internals en fonction de la proximité
        unassigned = [tid for tid in current_internals if tid not in new_mapping]
        for tid in unassigned:
            if self.freed_ids:
                # si plusieurs IDs libres, choisir le plus proche
                best_id = None
                best_dist = float('inf')
                cx, cy = centroids.get(tid, (None, None))
                # si pas de position, fallback au premier libre
                if cx is None:
                    aid = self.freed_ids.popleft()
                else:
                    for aid_candidate in list(self.freed_ids):
                        last_pos = self.last_positions.get(aid_candidate)
                        if last_pos:
                            dist = (cx - last_pos[0])**2 + (cy - last_pos[1])**2
                        else:
                            dist = float('inf')
                        if dist < best_dist:
                            best_dist = dist
                            best_id = aid_candidate
                    if best_id is None:
                        aid = self.freed_ids.popleft()
                    else:
                        aid = best_id
                        self.freed_ids.remove(best_id)
                new_mapping[tid] = aid
            else:
                # plus d'IDs disponibles, recycle circulaire
                new_mapping[tid] = ((tid - 1) % len(self.pool_ids)) + 1

        # 4) mettre à jour last_positions pour tous
        for tid, aid in new_mapping.items():
            if tid in centroids:
                self.last_positions[aid] = centroids[tid]

        # 5) sauvegarder mapping et internals pour next frame
        self.internal_to_assigned = new_mapping
        self.prev_internals = set(current_internals)
        return new_mapping


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
