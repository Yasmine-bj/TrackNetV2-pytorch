import requests
from constants.config import API_BASE_URL

def update_match_state(match_id, state, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(f"{API_BASE_URL}/admin/matches/{match_id}/state", json={"state": state}, headers=headers)
    return response.ok

def send_player_stats(match_id, stats_payload, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_BASE_URL}/admin/matches/{match_id}/participations/stats", json={"stats": stats_payload}, headers=headers)
    return response.ok
