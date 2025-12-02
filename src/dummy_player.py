from random import uniform, choice

from src.utils import *
from src.game import *
from src.db import dbHandler


class dummyPlayer:
    
    def __init__(self):
        self.db = dbHandler(TRAINING_DATA_PATH_1PLAYER)
        self.game = Game(self.db, debug=False)
        
    def play_rounds(self, rounds: int = 5, batch_size: int = 30) -> None:
        for _ in range(rounds):
            counter = 0
            while self.game.flag_won is None:
                best_action = [-100, 1, 0, 0]
                for _ in range(batch_size):
                    angle = uniform(0, 1)
                    power = uniform(0.5, 1)
                    ball = choice([b.index for b in self.game.balls if b.index != 0 and b.active])
                    score = self.game.simulate(ball, angle, power, backtrack=True)
                    if score > best_action[0]:
                        best_action = [score, ball, angle, power]
                self.game.simulate(best_action[1], best_action[2], best_action[3])
                counter += 1
                print(f"iter {counter}")
            print(f"Moves: {counter}")
            self.game.save_history()
            self.game = Game(self.db, debug=False)