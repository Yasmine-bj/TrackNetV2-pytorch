#!/usr/bin/env python
"""
Pipeline vidéo : YOLO ➜ ByteTrack ➜ classification zones ➜ export CSV + vidéo.
"""

from __future__ import annotations
import os
import cv2
import supervision as sv

from constants.config import (
    MODEL_PATH, VIDEO_IN, VIDEO_OUT, CSV_OUT, KPI_OUT,
    CONF_THRES, TERRAIN_POLYGON, ATTACK_ZONES,
)

from tracker.detector import YOLODetector
from tracker.byte_tracker import ByteTrackerWrapper
from tracker.annotators import create_annotators

from analytics.position_recorder import PositionRecorder
from analytics.role_classifier import RoleClassifier
from kpi.exporter import CSVExporter
from kpi.calculator import KPICalculator


def main() -> None:
    cap = cv2.VideoCapture(str(VIDEO_IN))
    if not cap.isOpened():
        raise SystemExit(f"Impossible d’ouvrir {VIDEO_IN}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(VIDEO_OUT), fourcc, 30.0, (width, height))

    detector  = YOLODetector(str(MODEL_PATH), conf=CONF_THRES)
    tracker   = ByteTrackerWrapper()
    box_annot, label_annot, *_ = create_annotators()
    role_clf = RoleClassifier(ATTACK_ZONES)
    recorder  = PositionRecorder()

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- détection puis filtrage au sein du terrain ---
        det_res    = detector.detect(frame)
        detections = sv.Detections.from_ultralytics(det_res)

        if detections.xyxy.size:
            valid = [
                i
                for i, (x1, y1, x2, y2) in enumerate(detections.xyxy)
                if cv2.pointPolygonTest(TERRAIN_POLYGON, (int(x1), int(y2)), False) >= 0
                and cv2.pointPolygonTest(TERRAIN_POLYGON, (int(x2), int(y2)), False) >= 0
            ]
            detections = detections[valid]

        detections = tracker.update(detections)

        # --- annotations ---
        labels     = [f"#{tid}" for tid in detections.tracker_id]
        annotated  = box_annot.annotate(frame.copy(), detections)
        annotated  = label_annot.annotate(annotated, detections, labels=labels)

        # --- classification attaque / défense ---
        for i in range(len(detections.xyxy)):
            x1, y1, x2, y2 = detections.xyxy[i]
            x,  y          = (x1 + x2) / 2, y2

            is_att, is_def, zone_id = role_clf.classify(detections.tracker_id[i], x, y)

            recorder.record(frame_id, detections.tracker_id[i], x, y, is_att, is_def)

            # texte sur la vidéo
            if is_att:
                cv2.putText(annotated, "Attaquant", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif is_def:
                cv2.putText(annotated, "Defenseur", (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,   0,255), 2)

        # polygone du terrain
        cv2.polylines(annotated, [TERRAIN_POLYGON], True, (0, 0, 255), 2)

        # --- dessiner les 4 zones d’attaque --------------------------
        # zone_colors = [(0,255,0), (255,255,0), (0,255,255), (255,0,255)]  # par ex.
        # for idx, zone in enumerate(ATTACK_ZONES):
            # cv2.polylines(annotated, [zone], True, zone_colors[idx], 2)

        

        out.write(annotated)
        frame_id += 1

    # --- export ---
    cap.release()
    out.release()

    # --- export positions ---
    df_positions = recorder.to_dataframe()
    CSVExporter(CSV_OUT).export(df_positions)

    # --- calcul & export KPI ---
    kpi_list = list(KPICalculator(df_positions).compute_percentages())
    CSVExporter(KPI_OUT).export(kpi_list)

    print("\n=== KPI Attaque / Défense ===")
    for k in kpi_list:
        print(f"Joueur {k['player_id']:>2} : "
            f"attaque {k['attack_pct']:.2f}% | "
            f"défense {k['defense_pct']:.2f}%")

    print(f"\nVidéo annotée   : {VIDEO_OUT}")
    print(f"Positions CSV   : {CSV_OUT}")
    print(f"KPI résumé CSV  : {KPI_OUT}")



if __name__ == "__main__":
    main()
