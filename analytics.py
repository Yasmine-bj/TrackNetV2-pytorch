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
    Associe une zone d'attaque à chaque joueur, puis détermine attaque vs défense.
    """
    def __init__(self):
        self.zones = ATTACK_ZONES
        self.player_zone: dict[int, int|None] = {}

    def _in_zone(self, idx: int, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.zones[idx], (int(x), int(y)), False) >= 0

    def classify(self, player_id: int, x: float, y: float) -> tuple[bool,bool,int|None]:
        if player_id not in self.player_zone:
            self.player_zone[player_id] = None
        zone = self.player_zone[player_id]

        if zone is None:
            for i in range(len(self.zones)):
                if self._in_zone(i, x, y):
                    self.player_zone[player_id] = i
                    zone = i
                    break

        is_attack = zone is not None and self._in_zone(zone, x, y)
        is_defense = not is_attack
        return is_attack, is_defense, zone

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
