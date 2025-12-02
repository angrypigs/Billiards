import sqlite3
from src.utils import * 
import numpy as np



class dbHandler:
    
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.cur = self.conn.cursor()
        
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {",".join([f"{x} REAL" for x in COLUMN_NAMES])},
                used INTEGER DEFAULT 0
            )
        """)
        
    def insert(self, data: list[float]) -> None:
        """
        Data order:
        - pairs of coords (x_0, y_0, x_1 ...) (cue ball first then regular ones) [32 pól]
        - target_idx [1 pole]
        - delta_angle_norm [1 pole]
        - power_norm [1 pole]
        - score [1 pole]
        """

        self.cur.execute(f"""INSERT INTO shots ({','.join(COLUMN_NAMES)}) 
                             VALUES ({",".join(["?" for _ in COLUMN_NAMES])})""",
                         data)
        self.conn.commit()
        
    def get_learning_data(self):
        self.cur.execute("SELECT * FROM shots WHERE used = 0")
        rows = self.cur.fetchall()
        if not rows:
            return None, None, None
        ids = []
        X_list = []
        y_list = []
        rewards = []

        for row in rows:
            rec_id, *coords_and_rest = row
            coords = coords_and_rest[:32]

            target_idx = coords_and_rest[32]
            delta_angle_norm = coords_and_rest[33]
            power_norm = coords_and_rest[34]
            score = coords_and_rest[35]
            
            ids.append(rec_id)
            coords_norm = []

            for i in range(0, 32, 2):
                x = coords[i]
                y = coords[i + 1]
                if x < 0 or y < 0:
                    coords_norm.extend([-1.0, -1.0])
                else:
                    coords_norm.append(x / WIDTH)
                    coords_norm.append(y / HEIGHT)
            X_list.append(coords_norm)
            y_list.append([target_idx, delta_angle_norm, power_norm]) 
            rewards.append(score)
            
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32) 
        rewards = np.array(rewards, dtype=np.float32)
        id_tuple = tuple(ids)
        if id_tuple:
            self.cur.execute(f"UPDATE shots SET used = 1 WHERE id IN ({','.join(['?']*len(ids))})", id_tuple)
            self.conn.commit()
        return X, y, rewards
        
        
    def close(self) -> None:
        self.conn.close()
