import pygame
import sys
import random

from src.utils import *
from src.game import Game
from src.db import dbHandler
from src.ai_player import AIPlayer

class Window:

    def __init__(self, agent_mode: bool = False, special_mode: int = 0) -> None:
        self.agent_mode = agent_mode
        self.special_mode = special_mode
        self.db = dbHandler(TRAINING_DATA_PATH_1PLAYER)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Billiards AI Showcase")
        init_assets()
        self.clock = pygame.time.Clock()
        self.game = Game(self.db, self.screen, debug=(not agent_mode), special_mode=special_mode)
        
        if agent_mode: 
            self.agent = AIPlayer()
            
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if not self.agent_mode:
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        self.game.release()

                    elif event.type == pygame.MOUSEMOTION:
                        if event.buttons[0]: 
                            self.game.load(event.pos)
                        else:
                            self.game.move(event.pos)

            if self.agent_mode:
                if self.game.player_flag is not None and not any([b.moving for b in self.game.balls]):
                    decision = self.agent.predict(self.game)
                    if decision:
                        idx, angle, power = decision
                        final_angle, final_power = self.game.agent_data_to_input(idx, angle, power)
                        print(f"Math: {idx} | AI Norm: {angle:.4f}")
                        self.game.shoot(final_angle, final_power) 
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

            self.game.draw()

            if self.game.flag_won is not None:
                self.game = Game(self.db, self.screen, special_mode=self.special_mode)
                
            pygame.display.flip()
            self.clock.tick(FPS)