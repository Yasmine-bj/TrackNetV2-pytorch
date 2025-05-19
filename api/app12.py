from flask import Flask, request, jsonify
import logging
import os
import csv
from datetime import datetime
from Enregistrement.enregistrement import record_http_stream, process_ia

app = Flask(__name__)

# Config logger
logging.basicConfig(level=logging.INFO)

@app.route('/start-match', methods=['POST'])
def start_match():
    data = request.get_json()

    match_id = data.get('match_id')
    rtsp_url = data.get('rtsp_url')
    duration = data.get('duration', 60)  # durée par défaut : 60 sec pour test

    if not match_id or not rtsp_url:
        return jsonify({"error": "Missing required fields: match_id or rtsp_url"}), 400

    output_dir = "Enregistrement/recordings"
    os.makedirs(output_dir, exist_ok=True)

    # Lancer l'enregistrement
    logging.info(f"[{match_id}] Starting recording from {rtsp_url}")
    success, thread = record_http_stream(match_id, duration, rtsp_url, output_dir)

    if success:
        thread.join()
        # video_path = os.path.join(output_dir, f"{match_id}.mp4")
        video_path ="input/this.mp4"
        logging.info(f"[{match_id}] Recording finished. Video saved: {video_path}")

        # Lancer le traitement IA
        logging.info(f"[{match_id}] Launching IA pipeline.")
        process_ia(video_path)

        # Lire le fichier CSV et construire la réponse JSON
        csv_file = "output/csv/kpi_summary.csv"
        if not os.path.exists(csv_file):
            return jsonify({"error": "kpi_summary.csv not found"}), 500

        result_data = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Skip lines where player_id is empty or non-numeric
                if not row['player_id'].isdigit():
                    continue
                result_data.append({
                    'player_id': row['player_id'],
                    'attack_pct': float(row['attack_pct']),
                    'defense_pct': float(row['defense_pct'])
                })


        return jsonify({
            "message": f"Match {match_id} processed successfully",
            "kpi_summary": result_data
        }), 200

    else:
        return jsonify({"error": f"Recording failed for {match_id}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1500)
