#!/usr/bin/env python
import os, cv2, time, threading, queue
import supervision as sv

from constants.config import (
    VIDEO_IN, VIDEO_OUT,
    CSV_OUT, KPI_OUT, BALL_CSV_OUT
)
from tracker import (
    YOLODetector, ByteTracker, Annotators,
    filter_on_terrain, TrackIDManager
)
from analytics import (
    PositionRecorder, RoleClassifier, BallRecorder
)
from kpi import KPI, CSVExporter
from tracknet_wrapper import TrackNetWrapper


def main():

    # ------------- ouverture vidéo + encodeur ---------------------------
    cap = cv2.VideoCapture(str(VIDEO_IN))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(VIDEO_OUT.parent, exist_ok=True)
    writer = cv2.VideoWriter(
        str(VIDEO_OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    # ------------- thread d’écriture vidéo (queue non bloquante) -------
    vqueue = queue.Queue(maxsize=16)

    def writer_thread():
        while True:
            fr = vqueue.get()
            if fr is None:             # sentinel de fin
                vqueue.task_done()     # ← ajoute cette ligne
                break
            writer.write(fr)
            vqueue.task_done()
        writer.release()


    threading.Thread(target=writer_thread, daemon=True).start()

    # ---------------- initialisation modules ---------------------------
    det   = YOLODetector()
    trk   = ByteTracker()
    ann   = Annotators()

    clf   = RoleClassifier()
    prec  = PositionRecorder()
    brec  = BallRecorder()
    idmgr = TrackIDManager(pool_size=4)

    tnet  = TrackNetWrapper()                 # TrackNet sur GPU

    frame_id   = 0
    total_start = time.time()

    # ---------------------- boucle vidéo -------------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- joueurs --------------------------------------------------
        res  = det.detect(frame)
        dets = sv.Detections.from_ultralytics(res)
        if dets.xyxy.size:
            dets = filter_on_terrain(dets)
        dets = trk.update(dets)

        centroids = {tid: ((x1+x2)/2, y2)
                     for (x1,y1,x2,y2), tid in zip(dets.xyxy, dets.tracker_id)}

        mapping  = idmgr.update(dets.tracker_id, centroids)
        labels   = [f"#{mapping[tid]}" for tid in dets.tracker_id]
        annotated = ann.apply(frame.copy(), dets, labels)

        # ---- balle ----------------------------------------------------
        bx, by, bvis = tnet.update(frame)
        if bvis:
            cv2.circle(annotated, (bx, by), 6, (0,255,255), -1)
        annotated = tnet.draw_traj(annotated)
        brec.record(frame_id, bx, by, bvis)

        # ---- enregistrement joueurs ----------------------------------
        for tid in dets.tracker_id:
            aid   = mapping[tid]
            x, y  = centroids[tid]
            is_a, is_d, _ = clf.classify(tid, x, y)
            prec.record(frame_id, aid, x, y, is_a, is_d)
            txt = f"ID{aid}:{'Attaque' if is_a else 'Defense'}"
            col = (0,255,0) if is_a else (0,0,255)
            cv2.putText(annotated, txt, (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        # ---- push frame -> queue --------------------------------------
        try:
            vqueue.put_nowait(annotated)
        except queue.Full:
            pass

        frame_id += 1

    # ---------------- nettoyage & export ------------------------------
    cap.release()
    vqueue.put(None)
    vqueue.join()

    print(f"Processing time: {time.time()-total_start:.2f}s")

    CSVExporter(CSV_OUT).export(prec.to_dataframe())
    CSVExporter(KPI_OUT).export(KPI.calculate(prec.to_dataframe()))
    CSVExporter(BALL_CSV_OUT).export(brec.to_dataframe())

    print(f"Vidéo annotée : {VIDEO_OUT}")
    print(f"CSV joueurs   : {CSV_OUT}")
    print(f"CSV balle     : {BALL_CSV_OUT}")
    print(f"KPI résumé    : {KPI_OUT}")


if __name__ == "__main__":
    main()
