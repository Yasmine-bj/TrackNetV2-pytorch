import sys
import os


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

@app.route('/start-match', methods=['POST'])
def start_match():
    data = request.get_json()

    match_id = data.get('match_id')
    rtsp_url = data.get('rtsp_url')
    duration = data.get('duration', 60)
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
        process_ia(video_path)

        # 6. Update state → done
        token = auth_manager.get_token()
        update_match_state(match_id, "done", token)

        # 7. Extraire les stats et envoyer à l’API
        token = auth_manager.get_token()
        try:
            kpi_file = "output/csv/kpi_summary.csv"
            with open(kpi_file, newline='') as f:
                reader = csv.DictReader(f)
                position_to_participation = {
                    int(p["position"]): p["participation_id"] for p in players
                }
                stats_payload = []
                for row in reader:
                    pos_id = int(row["player_id"])
                    participation_id = position_to_participation.get(pos_id)
                    if not participation_id:
                        continue
                    stats_payload.append({
                        "participation_id": participation_id,
                        "attack_pct": float(row["attack_pct"]),
                        "defense_pct": float(row["defense_pct"]),
                        "num_faults": 0,
                        "position_movement": []
                    })
            send_player_stats(match_id, stats_payload, token)
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi des stats: {e}")

        # 8. Nettoyage
        if os.path.exists(json_path):
            os.remove(json_path)
            logging.info("🧹 Fichier temporaire id_to_name.json supprimé.")

        return jsonify({"message": f"Match {match_id} processed successfully"}), 200

    else:
        return jsonify({"error": f"Recording failed for {match_id}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1500)
