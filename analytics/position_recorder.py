from __future__ import annotations
import pandas as pd

class PositionRecorder:
    """
    Accumule les positions + états attaque/défense
    pour export ultérieur en DataFrame.
    """

    def __init__(self) -> None:
        self._rows: list[dict] = []

    def record(
        self,
        frame_num: int,
        player_id: int,
        x: float,
        y: float,
        attack: bool,
        defense: bool,
    ) -> None:
        self._rows.append(
            {
                "frame_num": frame_num,
                "player_id": player_id,
                "x": x,
                "y": y,
                "attack": int(attack),
                "defense": int(defense),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)
