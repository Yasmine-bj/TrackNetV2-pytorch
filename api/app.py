from flask import Flask, request, jsonify
import logging
import os
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
    duration = data.get('duration', 60)  # durée par défaut : 1h30

    # Vérification stricte : on enlève players pour le moment
    if not match_id or not rtsp_url:
        return jsonify({"error": "Missing required fields: match_id or rtsp_url"}), 400

    output_dir = "Enregistrement/recordings"
    os.makedirs(output_dir, exist_ok=True)

    # Lancer l'enregistrement
    logging.info(f"[{match_id}] Starting recording from {rtsp_url}")
    success, thread = record_http_stream(match_id, duration, rtsp_url, output_dir)

    if success:
        thread.join()
        video_path = os.path.join(output_dir, f"{match_id}.mp4")
        logging.info(f"[{match_id}] Recording finished. Video saved: {video_path}")

        # Appel simplifié du traitement IA sans players
        logging.info(f"[{match_id}] Launching IA pipeline.")
        process_ia(video_path)

        return jsonify({"message": f"Match {match_id} processed successfully"}), 200
    else:
        return jsonify({"error": f"Recording failed for {match_id}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
