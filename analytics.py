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

        
class RoleClassifier:
    """
    Associe directement chaque joueur à sa zone d'attaque (selon son ID),
    puis détermine s'il est en attaque ou en défense.
    """
    def __init__(self):
        self.zones = ATTACK_ZONES  # liste : [ATTACK_ZONE_1, ATTACK_ZONE_2, ATTACK_ZONE_3, ATTACK_ZONE_4]

    def _in_zone(self, polygon, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

    def classify(self, player_id: int, x: float, y: float) -> tuple[bool, bool, int | None]:
        # Déduire la zone d’attaque directement depuis l’ID joueur
        if player_id < 1 or player_id > len(self.zones):
            # ID invalide
            return False, True, None

        attack_zone_idx = player_id - 1  # correspondance : ID 1 → index 0, ID 2 → index 1, etc.
        attack_zone = self.zones[attack_zone_idx]

        is_attack = self._in_zone(attack_zone, x, y)
        is_defense = not is_attack

        return is_attack, is_defense, attack_zone_idx


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
