import pygame

import sys

from src.utils import *
from src.game import Game

class Window:

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Billiards")
        init_assets()
        self.clock = pygame.time.Clock()
        self.game = Game(self.screen)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.game.release()
            if pygame.mouse.get_pressed()[0]:
                self.game.load(pygame.mouse.get_pos())
            self.game.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
        