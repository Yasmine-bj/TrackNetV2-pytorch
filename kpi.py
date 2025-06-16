class KPI:
    @staticmethod
    def calculate(df, faults_per_player=None):
        """
        faults_per_player : dict {player_id: nb_faults}
        """
        total = len(df["frame_num"].unique())
        result = []
        for pid, pdata in df.groupby("player_id"):
            row = {
                "player_id":  pid,
                "attack_pct": pdata["attack"].sum()  / total * 100,
                "defense_pct":pdata["defense"].sum() / total * 100,
            }
            # nouvelle colonne si le dict est fourni
            if faults_per_player and pid in faults_per_player:
                row["fault_filet"] = faults_per_player[pid]
            result.append(row)
        return result




class CSVExporter:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
    def export(self, data):
        import pandas as pd
        df = data if hasattr(data, 'to_csv') else pd.DataFrame(data)
        df.to_csv(self.path, index=False)