import os, sys
import pygame
from src.table_pockets_render import POCKET_RADIUS, render_pockets, calculate_holes

WIDTH = 1700
HEIGHT = 1000
FPS = 120

MAX_POWER = 15
RADIUS = 20
THRESHOLD = 5
LIMIT = THRESHOLD + RADIUS
CUE_RADIUS = 80

HOLES = calculate_holes(width=WIDTH, height=HEIGHT-100)

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

COLORS = [
    (255, 255, 255),
    (100, 100, 100),
    (229, 0, 0),
    (0, 229, 0),
    (0, 0, 229),
    (229, 229, 0),
    (229, 128, 0),
    (128, 0, 229),
    (0, 0, 0),
    (229, 0, 0),
    (0, 229, 0),
    (0, 0, 229),
    (229, 229, 0),
    (229, 128, 0),
    (128, 0, 229),
    (0, 229, 229),
]

def res_path(rel_path: str) -> str:
    """
    Return path to file modified by auto_py_to_exe path if packed to exe already
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = sys.path[0]
    return os.path.normpath(os.path.join(base_path, rel_path))

CSV_TRAINING_DATA_PATH = res_path("assets/training_data/data.csv")

def estimate_normal(mask: pygame.mask, cx: int, cy: int) -> tuple[int, int]:
    gx = mask.get_at((min(cx + 1, mask.get_size()[0] - 1), cy)) - mask.get_at((max(cx - 1, 0), cy))
    gy = mask.get_at((cx, min(cy + 1, mask.get_size()[1] - 1))) - mask.get_at((cx, max(cy - 1, 0)))
    length = (gx**2 + gy**2)**0.5
    if length == 0:
        return (0, -1)
    return (gx / length, gy / length)

IMAGES = {}

def init_assets() -> None:
    IMAGES["table"] = pygame.image.load("assets/textures/table.png").convert_alpha()
    IMAGES["table_bg"] = pygame.image.load("assets/textures/table_bg.png").convert_alpha()
    IMAGES["cue"] = pygame.Surface((100, 100), pygame.SRCALPHA)
    pygame.draw.rect(IMAGES["cue"], (0, 0, 0), (0, 40, 100, 20))