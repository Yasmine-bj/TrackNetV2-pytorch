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
    CSV_OUT, KPI_OUT, BALL_CSV_OUT,attack_zone_colors,TERRAIN_POLYGON,ATTACK_ZONES,NET_POLY
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
import logging
import asyncio
from constants.config import (
    ZONE_POLYGONS
)
from fault_net import FaultNetDetector


def format_time(frame_id, fps):
    """Convert frame number to MM:SS format."""
    total_seconds = int(frame_id / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02}:{seconds:02}"

def main():

    # Lire le mapping des joeurs depuis le fichier temporaire
    id_to_name = {}
    if os.path.exists("id_to_name.json"):
        with open("id_to_name.json", "r") as f:
            id_to_name = json.load(f)

    
    if len(sys.argv) > 2:
        video_path = sys.argv[1]
        output_path = sys.argv[2]
    else:
        video_path = str(VIDEO_IN)
        output_path = str(VIDEO_OUT)

    # ------------- ouverture vidéo + encodeur ---------------------------
    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )

    # ------------- thread d'écriture vidéo (queue non bloquante) -------
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


     # On garde la référence au thread pour pouvoir .join() ensuite
    writer_thr = threading.Thread(target=writer_thread, daemon=True)
    writer_thr.start()

    # ---------------- initialisation modules ---------------------------
    det   = YOLODetector()
    trk   = ByteTracker()
    ann   = Annotators()

    clf   = RoleClassifier()
    prec  = PositionRecorder()
    brec  = BallRecorder()
    idmgr = TrackIDManager(
    zone_polygons=ZONE_POLYGONS,
    assign_delay=5.0,
    grace_period= 11.0  ,
    reassign_dist=250
    )

    tnet = TrackNetWrapper()     # TrackNet sur GPU

    # ---- Fault-Net detector --------------------------------------
    fault_det = FaultNetDetector(
        net_poly=NET_POLY,
        fps=fps,
    )


 


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

            # C'est une compréhension de dictionnaire ({clé: valeur for ...})
            # qui construit un dictionnaire centroids où :

            # clé = tid → l'ID interne attribué par le tracker (par exemple : 5, 7, 12…)

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

        is_fault = fault_det.update(frame_id, bx, by, visible=bool(bvis))

        # (A) trajectoire jaune
        annotated = tnet.draw_traj(annotated, (0,255,255))

        # (B) segments rouges actifs
        annotated = fault_det.draw_fault_segments(annotated)

        # (C) point courant
        if bvis:
            col_pt = (0,0,255) if is_fault else (0,255,255)
            cv2.circle(annotated, (bx,by), 8, col_pt, -1)


        # (D) texte de la balle
        brec.record(frame_id, bx, by, bvis)



        # ---- enregistrement joueurs ----------------------------------
        for tid in dets.tracker_id:
            aid   = mapping[tid]  #assigned_id (zone index ou None)
            x, y  = centroids[tid]
            is_a, is_d, _ = clf.classify(aid , x, y)
            prec.record(frame_id, aid, x, y, is_a, is_d)
           
            #txt = f"ID{aid}:{'Attaquant' if is_a else 'Défenseur' if is_d else 'Inconnu'}"
            txt = f"ID{aid}:{'Attaquant' if is_a else ('Defenseur' if is_d else '')}"

            if is_a:
                col = (0, 255, 0)   # vert pour attaquant
            elif is_d:
                col = (0, 0, 255)   # rouge pour défenseur
            else:
                col = (0, 0, 0)     # noir si inconnu ou None
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

    
    # 1) On envoie le sentinel pour terminer le writer
    vqueue.put(None)
    # 2) On attend que toutes les frames soient écrites
    vqueue.join()
    # 3) On attend la fin du thread d'écriture
    writer_thr.join()
    # 4) On libère la capture
    cap.release()

    # ---------------- Appload sans S3------------------------------

    logging.info("Lancement de l'upload vers S3…")
    upload_url = s3_handler.upload_file(output_path)

    # ...après upload_url = s3_handler.upload_file(str(VIDEO_OUT))
    if upload_url:
        print(f"✅ Vidéo uploadée avec succès : {upload_url}")
        # Sauvegarder l'URL dans un fichier temporaire
        with open("output/video_url.txt", "w") as f:
            f.write(upload_url)
    else:
        print("❌ Échec de l'upload vers S3.")
  


    # # 4) Traitement du résultat
    # if upload_url:
    #     print(f"✅ Vidéo uploadée avec succès : {upload_url}")
    # else:
    #     print("❌ Échec de l'upload vers S3.")


    # ---------------- nettoyage & export ------------------------------

    print(f"Processing time: {time.time()-total_start:.2f}s")

    # ------ fautes filet -----------------------------------------
    team1_faults, team2_faults = fault_det.summary()
    print(f"Fautes filet  Équipe1(ID1-2)={team1_faults}  Équipe2(ID3-4)={team2_faults}")

    # dictionnaire joueur → nb de fautes
    faults_per_player = {
        1: team1_faults,
        2: team1_faults,
        3: team2_faults,
        4: team2_faults,
}



    CSVExporter(CSV_OUT).export(prec.to_dataframe())
    kpi_rows = KPI.calculate(prec.to_dataframe(), faults_per_player)
    CSVExporter(KPI_OUT).export(kpi_rows)
    CSVExporter(BALL_CSV_OUT).export(brec.to_dataframe())

    print(f"Vidéo annotée : {VIDEO_OUT}")
    print(f"CSV joueurs   : {CSV_OUT}")
    print(f"CSV balle     : {BALL_CSV_OUT}")
    print(f"KPI résumé    : {KPI_OUT}")


if __name__ == "__main__":
    main()
