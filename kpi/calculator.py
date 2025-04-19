class KPICalculator:
    """Pourcentage de présence en attaque / défense par joueur."""

    def __init__(self, df):
        self.df = df

    def compute_percentages(self):
        total_frames = len(self.df["frame_num"].unique())
        for player_id, pdata in self.df.groupby("player_id"):
            yield {
                "player_id": player_id,
                "attack_pct": pdata["attack"].sum() / total_frames * 100,
                "defense_pct": pdata["defense"].sum() / total_frames * 100,
            }
