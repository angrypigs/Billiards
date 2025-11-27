import sqlite3

from src.utils import TRAINING_DATA_PATH_1PLAYER, COLUMN_NAMES


class dbHandler:
    
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.cur = self.conn.cursor()
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS shots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                {",".join([f"{x} REAL" for x in COLUMN_NAMES])}
            )
        """)
        
    def insert(self, data: list[float]) -> None:
        """
        Data order:
        - pairs of coords (x_0, y_0, x_1 ...) (cue ball first then regular ones)
        - angle
        - power
        - score
        """
        self.cur.execute(f"""INSERT INTO shots ({','.join(COLUMN_NAMES)}) 
                         VALUES ({",".join(["?" for _ in COLUMN_NAMES])})""",
                         data)
        self.conn.commit()
        
    def close(self) -> None:
        self.conn.close()

