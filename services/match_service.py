import requests
from constants.config import API_BASE_URL

def update_match_state(match_id, state, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(f"{API_BASE_URL}/admin/matches/{match_id}/state", json={"state": state}, headers=headers)
    return response.ok

def send_player_stats(match_id, stats_payload, token, video_url):
    import logging
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{API_BASE_URL}/admin/matches/{match_id}/participations/stats"
    logging.info(f"Envoi des stats pour match {match_id} à {url} : {stats_payload} avec video_url={video_url}")
    payload = {
        "stats": stats_payload,
        "video_url": video_url
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.ok:
        logging.info(f"Stats envoyées avec succès pour match {match_id}.")
    else:
        logging.error(f"Échec de l'envoi des stats pour match {match_id} : {response.status_code} - {response.text}")
    return response.ok
