import os, sys
import pygame

WIDTH = 1700
HEIGHT = 1000
FPS = 120

MAX_POWER = 15
RADIUS = 15
THRESHOLD = 5
LIMIT = THRESHOLD + RADIUS

START_POS = (
    (0, 0),
    (1.75, -1),
    (1.75, 1),
    (3.5, -2),
    (3.5, 0),
    (3.5, 2),
    (3.5, -2),
    (5.25, -3),
    (5.25, -1),
    (5.25, 1),
    (5.25, 3),
    (7, -4),
    (7, -2),
    (7, 0),
    (7, 2),
    (7, 4),
)

COLORS = (
    (28, 101, 123),
    (45, 28, 122),
    (122, 28, 100)
)

def res_path(rel_path: str) -> str:
    """
    Return path to file modified by auto_py_to_exe path if packed to exe already
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = sys.path[0]
    return os.path.normpath(os.path.join(base_path, rel_path))

IMAGES = {}

def init_assets() -> None:
    IMAGES["table"] = pygame.image.load("assets/textures/table.png").convert_alpha()
    IMAGES["table_bg"] = pygame.image.load("assets/textures/table_bg.png").convert_alpha()