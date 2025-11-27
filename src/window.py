import pygame

import sys

from src.utils import *
from src.game import Game
from src.db import dbHandler

class Window:

    def __init__(self) -> None:
        self.db = dbHandler(TRAINING_DATA_PATH_1PLAYER)
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Billiards")
        # render_pockets(with_holes=True)
        init_assets()
        self.clock = pygame.time.Clock()
        self.game = Game(self.screen, self.db)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game.save_history()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.game.release()
                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons == (0, 0, 0):
                        self.game.move(event.pos)
            if pygame.mouse.get_pressed()[0]:
                self.game.load(pygame.mouse.get_pos())
            self.game.draw()
            if self.game.flag_won is not None:
                self.game.save_history()  
                self.game = Game(self.screen, self.db)
            pygame.display.flip()
            self.clock.tick(FPS)
            
        
        