import pygame

import sys
import os
import csv

from src.utils import *
from src.game import Game

class Window:

    def __init__(self) -> None:
        self.check_csv_history()
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Billiards")
        init_assets()
        # render_pockets()
        self.clock = pygame.time.Clock()
        self.game = Game(self.screen)
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
            if self.game.flag_won:
                self.game.save_history()
                self.game = Game(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
            
    def check_csv_history(self) -> None:
        p = CSV_TRAINING_DATA_PATH
        headers = ["angle", "power"]
        for i in range(16):
            for j in "xy":
                headers.append(f"{i}_{j}")
        headers.append("score")
        err = ""
        if not os.path.isfile(p):
            err = "No data"
        else:
            with open(p, newline="", encoding='utf-8') as f:
                reader = csv.reader(f)
                try:
                    actual_headers = next(reader)
                    if actual_headers != headers:
                        err = f"Invalid headers: {actual_headers} (should be {headers})"
                except StopIteration:
                    err = "Empty data - no headers"
        if err:
            print(f"Reading data.csv: {err}")
            with open(p, "w", newline="", encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        
        