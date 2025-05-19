#!/usr/bin/env python

from dotenv import load_dotenv
load_dotenv()

import os
from constants.config import Config
from storage.s3_handler import S3Handler

import os, cv2, time, threading, queue
import supervision as sv

from constants.config import (
    VIDEO_IN, VIDEO_OUT,
    CSV_OUT, KPI_OUT, BALL_CSV_OUT,attack_zone_colors,TERRAIN_POLYGON,ATTACK_ZONES
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
import sys
import json


import asyncio




def main():

    # Lire le mapping des joeurs depuis le fichier temporaire
    id_to_name = {}
    if os.path.exists("id_to_name.json"):
        with open("id_to_name.json", "r") as f:
            id_to_name = json.load(f)


    # Vérifier si un chemin vidéo est passé en argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = str(VIDEO_IN)  # Utiliser la valeur par défaut si aucun argument n'est passé

    # ------------- ouverture vidéo + encodeur ---------------------------
    cap = cv2.VideoCapture(video_path)
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

    tnet = TrackNetWrapper()     # TrackNet sur GPU


    # Instanciation config + handler
    config = Config()
    s3_handler= S3Handler(config)
             

    frame_id   = 0
    total_start = time.time()

    # ---------------------- boucle vidéo -------------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        
        # Ajouter la ligne du terrain et des zones d'attaque à la fin
        for zone, color in zip(ATTACK_ZONES, attack_zone_colors):
            frame = ann.fill_polygons(frame, [zone], color=color)

        # ---- joueurs --------------------------------------------------
        res  = det.detect(frame)
        dets = sv.Detections.from_ultralytics(res)
        if dets.xyxy.size:
            dets = filter_on_terrain(dets)
        # // fusion des détections
        dets = trk.update(dets) 

            # C’est une compréhension de dictionnaire ({clé: valeur for ...})
            # qui construit un dictionnaire centroids où :

            # clé = tid → l’ID interne attribué par le tracker (par exemple : 5, 7, 12…)

            # valeur = un tuple ( (x1 + x2) / 2, y2 )

        centroids = {tid: ((x1+x2)/2, y2)
                     for (x1,y1,x2,y2), tid in zip(dets.xyxy, dets.tracker_id)}

        mapping  = idmgr.update(dets.tracker_id, centroids)
        # // on applique le mapping pour obtenir les IDs assignés
        if id_to_name:
            labels = []
            for tid in dets.tracker_id:
                assigned_id = mapping[tid]
                name = id_to_name.get(str(assigned_id), "")
                labels.append(f"#{assigned_id} {name}")
        else:
            labels = [f"#{mapping[tid]}" for tid in dets.tracker_id]


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
        
        
        # ---- Ajouter les lignes des polygones -------------------------
        


        # # Ajouter la ligne du terrain et des zones d'attaque à la fin
        # annotated = ann.draw_polygons(annotated, [TERRAIN_POLYGON], color=(0, 255, 0), thickness=3)
        # surface attaque
        # for zone, color in zip(ATTACK_ZONES, attack_zone_colors):
        #     annotated = ann.fill_polygons(annotated, [zone], color=color)

        

        # ---- push frame -> queue --------------------------------------
        try:
            vqueue.put_nowait(annotated)
        except queue.Full:
            pass

        frame_id += 1

    # ---------------- Appload sans S3------------------------------
    upload_url = asyncio.run(s3_handler.upload_file(str(VIDEO_OUT)))

    if upload_url:
        print(f"✅ Vidéo uploadée avec succès sur S3 : {upload_url}")
    else:
        print("❌ Échec de l'upload vers S3.")

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
