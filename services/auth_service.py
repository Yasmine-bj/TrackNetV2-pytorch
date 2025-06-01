import requests
import time 
from constants.config import API_BASE_URL, LOGIN_CREDENTIALS


class AuthManager:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.expires_at = 0  # timestamp in seconds
        self.login()  # initial login

    def login(self):
        response = requests.post(f"{API_BASE_URL}/auth/login", json=LOGIN_CREDENTIALS)
        if response.status_code == 200:
            print("🔑 Connexion réussie !")
            resp_json = response.json()
            # Accès à la bonne profondeur
            data = resp_json["data"]["data"]
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.expires_at = time.time() + data["expires_in"] - 30  # 30s de marge
        else:
            raise Exception(f"Login failed: {response.status_code} → {response.text}")
    def is_token_expired(self):
        return time.time() >= self.expires_at

    def get_token(self):
        if self.access_token is None or self.is_token_expired():
            print("🔄 Token expiré ou manquant, reconnexion...")
            self.login()
        return self.access_token
