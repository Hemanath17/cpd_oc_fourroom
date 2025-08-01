# core/logger.py
import os
import pandas as pd

class CSVLogger:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df = pd.DataFrame(columns=columns)

    def log_row(self, **kwargs):
        self.df.loc[len(self.df)] = [kwargs.get(c, None) for c in self.columns]

    def flush(self):
        self.df.to_csv(self.path, index=False)
