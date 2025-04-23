class KPI:
    @staticmethod
    def calculate(df):
        total = len(df["frame_num"].unique())
        return [
            {
                "player_id": pid,
                "attack_pct": pdata["attack"].sum()/total*100,
                "defense_pct": pdata["defense"].sum()/total*100,
            }
            for pid, pdata in df.groupby("player_id")
        ]

class CSVExporter:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def export(self, data):
        import pandas as pd
        df = data if hasattr(data, 'to_csv') else pd.DataFrame(data)
        df.to_csv(self.path, index=False)