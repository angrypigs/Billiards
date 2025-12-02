import pygame

import sys

from src.utils import *
from src.game import Game
from src.db import dbHandler
from src.agent_rl import RLAgent

class Window:

    def __init__(self, agent_mode: bool = False) -> None:
        self.agent_mode = agent_mode
        self.db = dbHandler(TRAINING_DATA_PATH_1PLAYER)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Billiards")
        # render_pockets(with_holes=True)
        init_assets()
        self.clock = pygame.time.Clock()
        self.game = Game(self.db, self.screen)
        if agent_mode: 
            self.agent = RLAgent(game=self.game)
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
                if self.game.player_flag is not None:
                    state = self.agent.get_state()
                    idx, angle, power = self.agent.predict(state, False)
                    new_angle, new_power = self.game.agent_data_to_input(idx, angle, power)
                    self.game.shoot(new_angle, new_power)
            else:
                if pygame.mouse.get_pressed()[0]:
                    self.game.load(pygame.mouse.get_pos())
            self.game.draw()
            if self.game.flag_won is not None:
                self.game = Game(self.db, self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
            
        
        