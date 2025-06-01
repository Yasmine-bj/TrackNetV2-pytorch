import cv2
import os
import time
import threading
import logging
from subprocess import run 

import cv2
import os
import time
import threading
import logging

def record_http_stream(match_id, duration_seconds, stream_url, output_dir):
    """
    Enregistre un flux HTTP vidéo en arrière-plan pendant une durée définie.
    
    Args:
        match_id (str): Identifiant du match (nom du fichier de sortie sans extension).
        duration_seconds (int): Durée maximale de l'enregistrement en secondes.
        stream_url (str): URL du flux HTTP.
        output_dir (str): Dossier de sortie pour enregistrer la vidéo.

    Returns:
        bool: True si l'enregistrement s'est lancé avec succès, False sinon.
    """

    output_path = os.path.join(output_dir, f"{match_id}.mp4")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        logging.error(f"[{match_id}] Impossible d'ouvrir le flux HTTP.")
        return False

    # Obtenir les propriétés de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    def _record():
        logging.info(f"[{match_id}] Début de l'enregistrement threadé pour {duration_seconds} secondes.")
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"[{match_id}] Problème lors de la lecture du flux. Arrêt de l'enregistrement.")
                break
            out.write(frame)
        cap.release()
        out.release()
        logging.info(f"[{match_id}] Enregistrement terminé. Fichier sauvegardé : {output_path}")

    # Lancer le thread
    thread = threading.Thread(target=_record, daemon=True)
    thread.start()

    return True, thread

def process_ia(video_path, output_video_path):
    """
    Appelle le pipeline IA pour traiter la vidéo enregistrée.
    
    Args:
        video_path (str): Chemin de la vidéo à traiter.
        output_video_path (str): Chemin de sortie de la vidéo annotée.
    """
    try:
        # Appeler le script principal de traitement IA avec deux arguments
        result = run(["python3", "/app/main.py", video_path, output_video_path], check=True)
        logging.info(f"Traitement IA terminé avec succès pour la vidéo : {video_path}")
    except Exception as e:
        logging.error(f"Erreur lors du traitement IA : {e}")