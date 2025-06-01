import os
import logging
from datetime import datetime
from enregistrement import record_http_stream,process_ia  
import threading

# suppose que la fonction est dans record_http.py

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def main():
    # Paramètres
    match_id = f"match_{datetime.now()}"
    duration_seconds = 20 # 1h30
    stream_url = "http://41.225.16.233:8080/terrain1"  # Remplace par ton vrai lien HTTP
    output_dir = "Enregistrement/recordings"

    # Créer le dossier de sortie s’il n’existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Lancer l'enregistrement
    logging.info(f"[{match_id}] Début de l'enregistrement du flux vidéo.")
    success, thread  = record_http_stream(match_id, duration_seconds, stream_url, output_dir)

    if success:
        thread.join() 
        # Attendre la fin de l'enregistrement (le thread s'exécute en arrière-plan)
        video_path = os.path.join(output_dir, f"{match_id}.mp4")
        logging.info(f"[{match_id}] Enregistrement terminé. Vidéo sauvegardée : {video_path}")

        # Lancer le traitement IA sur la vidéo enregistrée
        logging.info(f"[{match_id}] Lancement du traitement IA sur la vidéo.")
        
        
        process_ia(video_path, output_video_path)       
    else:
        logging.error(f"[{match_id}] Échec de l'enregistrement.")

if __name__ == "__main__":
    main()
