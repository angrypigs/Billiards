import pygame
import sys

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
        pygame.display.set_caption("Billiards")
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
                        if event.buttons == (0, 0, 0):
                            self.game.move(event.pos)
            
            if self.agent_mode:
                if self.game.player_flag is not None and not any([b.moving for b in self.game.balls]):
                    decision = self.agent.predict(self.game)
                    if decision:
                        idx, raw_angle, raw_power = decision

                        target_ball = next((b for b in self.game.balls if b.index == idx), None)
                        bounds = self.game.ball_angle_range(target_ball) if target_ball else None
                        
                        if bounds is None:
                            print(f"AI Error: Selected ball {idx} is blocked/invalid. Skipping shot.")
                            self.game.shoot(0, 5) 
                        else:
                            new_angle, new_power = self.game.agent_data_to_input(idx, raw_angle, raw_power)
                            
                            print(f"AI: Ball {idx} | Raw: {raw_angle:.2f} -> Deg: {new_angle:.1f}")
                            self.game.shoot(new_angle, new_power)
                    else:
                         self.game.shoot(0, 10)

            else:
                if pygame.mouse.get_pressed()[0]:
                    self.game.load(pygame.mouse.get_pos())
            
            self.game.draw()
            
            if self.game.flag_won is not None:
                self.game = Game(self.db, self.screen, special_mode=self.special_mode)
                
            pygame.display.flip()
            self.clock.tick(FPS)