import sys
import os
sys.path.append('/app')  # Add /app to the Python path

print("Python Path:", sys.path)  # Debug the Python path

from flask import Flask, request, jsonify
import logging
import csv
import json
from datetime import datetime
from Enregistrement.enregistrement import record_http_stream, process_ia

from services.auth_service import AuthManager
from services.match_service import update_match_state, send_player_stats



app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

auth_manager = AuthManager()  # Instance globale

print(sys.path)

@app.route('/start-match', methods=['POST'])
def start_match():
    data = request.get_json()
    logging.info(f"Requête reçue sur /start-match : {data}")  # Ajout du log ici
    match_id = data.get('match_id')
    rtsp_url = data.get('rtsp_url')
    duration = 20
    # duration = data.get('duration', 60)
    players = data.get('players', [])

    if not match_id or not rtsp_url:
        return jsonify({"error": "Missing required fields: match_id or rtsp_url"}), 400

    if len(players) != 4:
        return jsonify({"error": "Exactly 4 players must be provided."}), 400

    # 1. Sauver id_to_name.json
    id_to_name = {p["position"]: p["name"] for p in players}
    json_path = "id_to_name.json"
    with open(json_path, "w") as f:
        json.dump(id_to_name, f)

    # 2. Update match state → recording
    token = auth_manager.get_token()
    update_match_state(match_id, "recording", token)

    # 3. Record video
    output_dir = "Enregistrement/recordings"
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"[{match_id}] Starting recording from {rtsp_url}")
    success, thread = record_http_stream(match_id, duration, rtsp_url, output_dir)

    if success:
        thread.join()
        video_path = os.path.join(output_dir, f"{match_id}.mp4")
        logging.info(f"[{match_id}] Recording finished. Video saved: {video_path}")

        # 4. Update state → processing
        token = auth_manager.get_token()
        update_match_state(match_id, "processing", token)

        # 5. Run IA pipeline
        output_video_path = f"output/{match_id}.mp4"
        process_ia(video_path, output_video_path)

        # 6. Update state → done
        token = auth_manager.get_token()
        update_match_state(match_id, "done", token)


        # 7. Extraire les stats et envoyer à l'API
        token = auth_manager.get_token()
        csv_file = "output/csv/kpi_summary.csv"
        logging.info(f"[{match_id}] Lecture du fichier KPI: {csv_file}")
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            position_to_participation = {
                    int(p["position"]): p["participation_id"] for p in players
                }
            stats_payload = []
            # Create a set of all player positions from the players list
            all_positions = {int(p["position"]) for p in players}
            # Create a set of positions found in the KPI file
            found_positions = set()

            # First pass: Process the KPI file and collect found positions
            for row in reader:
                pos_id = int(float(row['player_id']))
                found_positions.add(pos_id)
                participation_id = position_to_participation.get(pos_id)
                stats = {
                        "participation_id": participation_id,
                    "attack_pct": {"Float64": float(row["attack_pct"]), "Valid": True},
                    "defense_pct": {"Float64": float(row["defense_pct"]), "Valid": True},
                    "num_faults": {"Int64": 0, "Valid": True},
                        "position_movement": []
                    }
                logging.info(f"[{match_id}] Statistiques extraites pour participation_id {participation_id}: {stats}")
                stats_payload.append(stats)

            # Second pass: Add default values for missing positions
            for pos_id in all_positions - found_positions:
                participation_id = position_to_participation.get(pos_id)
                stats = {
                    "participation_id": participation_id,
                    "attack_pct": {"Float64": 0.0, "Valid": True},
                    "defense_pct": {"Float64": 0.0, "Valid": True},
                    "num_faults": {"Int64": 0, "Valid": True},
                    "position_movement": []
                }
                logging.warning(f"[{match_id}] Player ID {pos_id} not found in KPIs. Sending default values.")
                stats_payload.append(stats)

            # ...avant send_player_stats(...)
            video_url = ""
            if os.path.exists("output/video_url.txt"):
                with open("output/video_url.txt", "r") as f:
                    video_url = f.read().strip()
            else:
                logging.warning("Aucune URL vidéo trouvée, video_url sera vide.")

            logging.info(f"[{match_id}] Statistiques prêtes à être envoyées: {stats_payload} (video_url={video_url})")
            send_player_stats(match_id, stats_payload, token, video_url)
      


        # 8. Nettoyage
        if os.path.exists(json_path):
            os.remove(json_path)
            logging.info("🧹 Fichier temporaire id_to_name.json supprimé.")

        if os.path.exists("output/video_url.txt"):
            os.remove("output/video_url.txt")
            logging.info("🧹 Fichier temporaire video_url.txt supprimé.")

        return jsonify({"message": f"Match {match_id} processed successfully"}), 200

    else:
        return jsonify({"error": f"Recording failed for {match_id}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1500)
