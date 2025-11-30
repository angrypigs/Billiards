from random import uniform

from src.utils import *
from src.game import *
from src.db import dbHandler


class dummyPlayer:
    
    def __init__(self):
        self.db = dbHandler(TRAINING_DATA_PATH_1PLAYER)
        self.game = Game(self.db, debug=False)
        
    def play_rounds(self, rounds: int = 5, batch_size: int = 20) -> None:
        for _ in range(rounds):
            counter = 0
            while self.game.flag_won is None:
                best_action = [-100, 0, 0]
                for _ in range(batch_size):
                    angle = uniform(-180, 180)
                    power = uniform(MAX_POWER / 4, MAX_POWER)
                    score = self.game.simulate(angle, power, backtrack=True)
                    if score > best_action[0]:
                        best_action = [score, angle, power]
                self.game.simulate(best_action[1], best_action[2])
                counter += 1
            print(f"Moves: {counter}")
            self.game.save_history()
            self.game = Game(self.db, debug=False)