import sqlite3
import numpy as np
from src.utils import *

class dbHandler:
    def __init__(self, db_path=TRAINING_DATA_PATH_1PLAYER):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self.create_table()

    def create_table(self):
        coords_cols = ", ".join([f"x_{i} REAL, y_{i} REAL" for i in range(16)])
        query = f"""
            CREATE TABLE IF NOT EXISTS shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {coords_cols},
                ball INTEGER, 
                angle REAL,
                score REAL
            )
        """
        self.cur.execute(query)
        self.conn.commit()

    def insert(self, data: list[float]) -> None:
        """
        Data order:
        - pairs of coords (x_0, y_0, x_1 ...) (cue ball first then regular ones) [32 pól]
        - target_idx [1 pole]
        - delta_angle_norm [1 pole]
        - score [1 pole]
        """

        self.cur.execute(f"""INSERT INTO shots ({','.join(COLUMN_NAMES)}) 
                             VALUES ({",".join(["?" for _ in COLUMN_NAMES])})""",
                         data)
        self.conn.commit()

    def get_learning_data(self, limit: int = 500000):
        query = f"SELECT * FROM shots WHERE score > 0 ORDER BY RANDOM() LIMIT {limit}"
        self.cur.execute(query)
        rows = self.cur.fetchall()
        
        if not rows:
            print("DB: No data found!")
            return None, None, None
            
        X_list = []
        y_list = []

        for row in rows:
            rec_id, *coords_and_rest = row
            coords = coords_and_rest[:32]

            target_idx = coords_and_rest[32]
            recorded_angle = coords_and_rest[33]

            if abs(recorded_angle) > 1.05:
                continue

            wx = coords[0]
            wy = coords[1]
            features = []
            features.extend([wx / WIDTH, wy / HEIGHT])

            for i in range(2, 32, 2):
                bx = coords[i]
                by = coords[i+1]
                if bx < 0:
                    features.extend([0.0, 0.0])
                else:
                    dx = bx - wx
                    dy = by - wy
                    dist = np.sqrt(dx**2 + dy**2) / DIAGONAL
                    angle = np.arctan2(dy, dx) / np.pi
                    
                    features.extend([dist, angle])
            
            X_list.append(features)
            y_list.append([target_idx, recorded_angle])
            
        if len(X_list) == 0:
            print("DB: Data filtered out completely!")
            return None, None, None

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        return X, y, None
    
    def close(self):
        self.conn.close()