import pandas as pd
import cv2
from constants.config import ATTACK_ZONES

class PositionRecorder:
    """
    Accumule les positions et états attaque/défense.
    """
    def __init__(self):
        self._rows: list[dict] = []

    def record(self, frame_num: int, player_id: int,
               x: float, y: float, attack: bool, defense: bool) -> None:
        self._rows.append({
            "frame_num": frame_num,
            "player_id": player_id,
            "x": x,
            "y": y,
            "attack": int(attack),
            "defense": int(defense),
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

import cv2

class RoleClassifier:
    """
    Associe chaque joueur à sa zone d'attaque (via assigned_id),
    puis détermine s'il est en attaque ou en défense selon sa position.
    """
    def __init__(self):
        """
        :param attack_zones: liste de polygones (zones d'attaque)
        """
        self.zones = ATTACK_ZONES  # liste : [ATTACK_ZONE_1, ATTACK_ZONE_2, ATTACK_ZONE_3, ATTACK_ZONE_4]

    def _in_zone(self, polygon, x: float, y: float) -> bool:
        """
        Teste si le point (x,y) est dans le polygone (zone).
        """
        return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

    def classify(self, assigned_id: int | None, x: float, y: float) -> tuple[bool, bool, int | None]:
        """
        :param assigned_id: index de zone assignée au joueur (0-based) ou None
        :param x: position x du joueur
        :param y: position y du joueur
        :return: (is_attack, is_defense, assigned_id)
        """
        if assigned_id is None:
            # Pas d'assignation → pas d'attaque, pas de défense, id None
            return False, False, None

        if assigned_id < 1 or assigned_id > len(self.zones):
            # ID hors limite → pas assigné à une zone valide
            return False, True, None

        attack_zone = self.zones[assigned_id-1]
        is_attack = self._in_zone(attack_zone, x, y)
        is_defense = not is_attack
        # print(f"[DEBUG classify] assigned_id={assigned_id}, pos=({x:.1f},{y:.1f}), is_attack={is_attack}, is_defense={is_defense}")

        return is_attack, is_defense, assigned_id

        

class BallRecorder:
    def __init__(self):
        self._rows = []
    def record(self, frame_num:int, x:int|None, y:int|None, vis:float):
        self._rows.append({
            "frame_num": frame_num,
            "x": x if vis else None,
            "y": y if vis else None,
            "visibility": vis
        })
    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self._rows)
