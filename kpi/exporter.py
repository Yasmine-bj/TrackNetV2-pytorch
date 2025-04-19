import pandas as pd

class CSVExporter:
    """Sauvegarde un DataFrame OU une liste de dicts dans un CSV."""

    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier

    def export(self, data):
        """
        - `data` peut être un DataFrame OU une liste de dicts.
        """
        if isinstance(data, pd.DataFrame):
            df = data
        else:                      # liste de dicts => DataFrame
            df = pd.DataFrame(data)
        df.to_csv(self.path, index=False)
