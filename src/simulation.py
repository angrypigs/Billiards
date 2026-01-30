import random

from src.utils import *
from src.game import Game
from src.db import dbHandler
from src.ai_player import AIPlayer


class AISimulation:
    
    def __init__(self):
        self.db = dbHandler()
        self.game = Game(self.db, off_screen=True, debug=False)
        self.agent = AIPlayer()
        
    def simulate_matches(self, rounds: int = 1) -> None:
        stats = []
        for _ in range(rounds):
            for i in range(1000):
                decision = self.agent.predict(self.game)
                if decision:
                    idx, angle, power = decision
                    self.game.simulate(idx, angle, power)
                else:
                    print("AI: Error of prediction: random shot")
                    active_balls = [b.index for b in self.game.balls if b.active and b.index != 0]
                    
                    if active_balls:
                        rand_idx = random.choice(active_balls)
                        rand_angle = random.uniform(-1.0, 1.0)
                        rand_power = 1.0 
                        self.game.simulate(rand_idx, rand_angle, rand_power)
                    else:
                        break
                if self.game.flag_won is not None:
                    print(f"Round finished after {i + 1} shots")
                    self.game.reset()
                    stats.append(i + 1)
                    break
            else:
                active_balls = [b.index for b in self.game.balls if b.active and b.index != 0]
                print(f"Round not finished (1k limit reached), balls left: {len(active_balls)}")
                stats.append(1000)
                self.game.reset()
        print("=" * 40)
        print("\nSimulation results\n")
        print("=" * 40)
        print(f"\nRounds: {rounds}\nAvg shots per round: {sum(stats) / len(stats)}")
        print(f"\nBest round: {min(stats)} shots\nWorst round: {max(stats)} shots\n")
        
        