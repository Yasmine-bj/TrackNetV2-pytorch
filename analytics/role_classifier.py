import cv2
from typing import List, Dict
from constants.config import ATTACK_ZONES

class RoleClassifier:
    """
    - À la première détection, un joueur n'a pas encore de zone (valeur None).
    - Tant qu'aucune zone n'est attribuée, il est considéré Défenseur.
    - Dès qu'il entre pour la première fois dans une zone d'attaque,
      cette zone est définitivement associée.
    - Il est Attaquant uniquement s'il se trouve DANS sa zone attribuée.
    """

    def __init__(self, zones: List) -> None:
        self.zones = zones
        self.player_zone: Dict[int, int | None] = {}   # None = pas encore attribuée

    def _in_zone(self, zone_idx: int, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.zones[zone_idx], (int(x), int(y)), False) >= 0

    def classify(self, player_id: int, x: float, y: float):
        # ---------- initialisation ----------
        if player_id not in self.player_zone:
            self.player_zone[player_id] = None   # aucune zone assignée pour l'instant

        zone_idx = self.player_zone[player_id]

        # ---------- attribution éventuelle ----------
        if zone_idx is None:                                   # pas encore d'affectation
            for idx in range(len(self.zones)):
                if self._in_zone(idx, x, y):                   # il vient d'entrer dans une zone
                    self.player_zone[player_id] = idx          # on l'assigne définitivement
                    zone_idx = idx
                    break                                      # on s'arrête à la première zone trouvée

        # ---------- rôle actuel ----------
        is_attack = zone_idx is not None and self._in_zone(zone_idx, x, y)
        is_defense = not is_attack

        return is_attack, is_defense, zone_idx     # zone_idx peut rester None
