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
    ATTACK_ZONES,
    ZONE_POLYGONS
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

    # methode pour mise a jour du tracker ( fusion des detections dans une frame dans un seul objet) 

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




# Gestion des IDs (pool fixe de 4 joueurs) avec proximité spatiale et premiere assignation selon zone 


class TrackIDManager:
    def __init__(self, pool_size=4, zone_polygons=None):
        self.pool_ids = list(range(1, pool_size + 1))
        self.internal_to_assigned = {}
        self.last_positions = {}
        self.freed_ids = deque()  # IDs libérés après disparition
        self.prev_internals = set()
        self.zone_polygons = zone_polygons or {}
        self.zone_assigned = set()  # garde trace des zones déjà attribuées
        self.initial_assignment_done = False

    def point_in_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def assign_zone_based_id(self, point):
        for zone_id, polygon in self.zone_polygons.items():
            if zone_id not in self.zone_assigned and self.point_in_polygon(point, polygon):
                return zone_id
        return None

    def update(self, current_internals, centroids):
        
        # Met à jour le mapping des IDs visibles (assigned IDs) pour les objets détectés à partir des IDs internes du tracker.

        # Flux détaillé :
        # 1️⃣ Libérer les IDs dont les objets ont disparu :
        #     - Compare les IDs internes de la frame précédente à ceux de la frame actuelle.
        #     - Les IDs internes disparus libèrent leurs assigned IDs, qui sont mis en attente dans `freed_ids`.

        # 2️⃣ Conserver les mappings déjà existants:
        #     - Pour les objets toujours présents, conserve l’assignation précédente sans modification.

        # 3️⃣ Assigner les IDs aux nouveaux objets détectés :
        #     a) Si c’est le premier passage (avant que les zones 1–4 soient toutes attribuées) :
        #         - Utilise les zones (polygones) pour assigner les IDs 1, 2, 3, 4
        #           selon où se trouve l’objet dans le terrain.
        #         - Marque chaque zone comme déjà attribuée dès qu’un ID est donné.

        #     b) Après la première attribution:
        #         - Si un seul ID est libre, on l’attribue directement au prochain nouvel objet.
        #         - Si plusieurs IDs sont libres, on calcule la distance entre les nouvelles détections et 
        #           les dernières positions connues, et on choisit l’ID le plus proche.
        #         - S’il n’y a plus d’ID libre, on recycle circulairement les IDs à l’aide d’un modulo.

        # 4️⃣ Mettre à jour les positions des assigned IDs:
        #     - Enregistre la position actuelle (centroïde) des objets pour le prochain calcul de proximité.

        # Retour:
        #     - Un dictionnaire `{ internal_id → assigned_id }` contenant les nouvelles associations
        #       pour tous les objets actifs dans la frame actuelle.
        

        new_mapping = {}
        disappeared = self.prev_internals - set(current_internals)

        # 1) Libérer les IDs pour les internes disparus
        for old in disappeared:
            assigned = self.internal_to_assigned.pop(old)
            self.last_positions[assigned] = self.last_positions.get(assigned)
            self.freed_ids.append(assigned)

        # 2) Conserver les mappings déjà existants
        for tid, aid in self.internal_to_assigned.items():
            new_mapping[tid] = aid

        # 3) Assigner les nouveaux
        unassigned = [tid for tid in current_internals if tid not in new_mapping]

        for tid in unassigned:
            cx, cy = centroids.get(tid, (None, None))

            # --- PREMIÈRE ASSIGNATION PAR ZONE ---
            if not self.initial_assignment_done:
                if cx is not None and cy is not None:
                    assigned_zone_id = self.assign_zone_based_id((cx, cy))
                    if assigned_zone_id:
                        new_mapping[tid] = assigned_zone_id
                        self.zone_assigned.add(assigned_zone_id)
                        if len(self.zone_assigned) == len(self.pool_ids):
                            self.initial_assignment_done = True
                        continue  # passe à la détection suivante

            # --- APRÈS PREMIÈRE ATTRIBUTION---
            if len(self.freed_ids) == 1:
                # Un seul ID libre → réutilise-le directement
                aid = self.freed_ids.popleft()
                new_mapping[tid] = aid
            elif len(self.freed_ids) > 1:
                # Plusieurs IDs libres → choisit le plus proche
                best_id = None
                best_dist = float('inf')
                if cx is not None:
                    for aid_candidate in list(self.freed_ids):
                        last_pos = self.last_positions.get(aid_candidate)
                        if last_pos:
                            dist = (cx - last_pos[0]) ** 2 + (cy - last_pos[1]) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_id = aid_candidate
                if best_id is not None:
                    self.freed_ids.remove(best_id)
                    new_mapping[tid] = best_id
                else:
                    # fallback si aucune position connue
                    aid = self.freed_ids.popleft()
                    new_mapping[tid] = aid
            else:
                # Pas d’IDs libres → recycle circulairement
                new_mapping[tid] = ((tid - 1) % len(self.pool_ids)) + 1

        # 4) Mettre à jour les positions connues
        for tid, aid in new_mapping.items():
            if tid in centroids:
                self.last_positions[aid] = centroids[tid]

        self.internal_to_assigned = new_mapping
        self.prev_internals = set(current_internals)
        return new_mapping



class Annotators:
    def __init__(self):
        from supervision import Color, BoxAnnotator, LabelAnnotator
        col = Color(0, 0, 0)
        self.box = BoxAnnotator(color=col, thickness=2)
        self.label = LabelAnnotator(color=col)

    def apply(self, frame, detections, labels):
        frame = self.box.annotate(frame, detections)
        return self.label.annotate(frame, detections, labels=labels)

    @staticmethod
    def draw_polygons(frame, polygons, color=(0, 255, 0), thickness=2):
        for polygon in polygons:
            pts = np.array(polygon, np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
        return frame

    @staticmethod
    def fill_polygons(frame, polygons, color=(0, 255, 0), alpha=0.4):
        overlay = frame.copy()
        for polygon in polygons:
            pts = np.array(polygon, np.int32)
            cv2.fillPoly(overlay, [pts], color=color)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def filter_on_terrain(detections: sv.Detections) -> sv.Detections:
        valid = []
        terrain_poly = np.array(TERRAIN_POLYGON, np.int32)
        for i, (x1, y1, x2, y2) in enumerate(detections.xyxy):
            if (cv2.pointPolygonTest(terrain_poly, (int(x1), int(y2)), False) >= 0
                and cv2.pointPolygonTest(terrain_poly, (int(x2), int(y2)), False) >= 0):
                valid.append(i)
        return detections[valid]
