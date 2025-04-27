
#!/usr/bin/env python
import os, cv2, supervision as sv
from constants.config import VIDEO_IN, VIDEO_OUT, CSV_OUT, KPI_OUT
from tracker import YOLODetector, ByteTracker, Annotators, filter_on_terrain,TrackIDManager
from analytics import PositionRecorder, RoleClassifier
from kpi import KPI, CSVExporter
import time

def main():
    cap = cv2.VideoCapture(str(VIDEO_IN))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(VIDEO_OUT.parent, exist_ok=True)
    out = cv2.VideoWriter(
        str(VIDEO_OUT), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )

    det = YOLODetector()
    trk = ByteTracker()
    ann = Annotators()
    clf = RoleClassifier()
    rec = PositionRecorder()
    id_manager = TrackIDManager(pool_size=4)

    frame_id = 0
    total_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Détection + fusion + tracking
        res  = det.detect(frame)
        dets = sv.Detections.from_ultralytics(res)
        if dets.xyxy.size:
            dets = filter_on_terrain(dets)
        dets = trk.update(dets)

        # calcul des centroïdes pour chaque tracker internal
        centroids = {}
        for i, tid in enumerate(dets.tracker_id):
            x1, y1, x2, y2 = dets.xyxy[i]
            centroids[tid] = ((x1 + x2) / 2, y2)

        # Réassignation des IDs par proximité
        mapping = id_manager.update(dets.tracker_id, centroids)
        labels = [f"#{mapping[tid]}" for tid in dets.tracker_id]

        # Annotation et enregistrement
        annotated = ann.apply(frame.copy(), dets, labels)
        for i, tid in enumerate(dets.tracker_id):
            assigned_id = mapping[tid]
            x, y = centroids[tid]
            is_att, is_def, _ = clf.classify(int(tid), x, y)
            rec.record(frame_id, assigned_id, x, y, is_att, is_def)
            txt = f"ID{assigned_id}:" + ("Attaque" if is_att else "Defense")
            col = (0,255,0) if is_att else (0,0,255)
            cv2.putText(
                annotated, txt,
                (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, col, 2
            )

        out.write(annotated)
        frame_id += 1

    cap.release()
    out.release()

    total_time = time.time() - total_start
    print(f"Total processing time: {total_time:.2f}s")

    # Export des données et KPI
    df = rec.to_dataframe()
    CSVExporter(CSV_OUT).export(df)
    stats = KPI.calculate(df)
    CSVExporter(KPI_OUT).export(stats)

    print(f"Vidéo annotée : {VIDEO_OUT}")
    print(f"Positions CSV : {CSV_OUT}")
    print(f"KPI résumé   : {KPI_OUT}")

if __name__ == "__main__":
    main()
