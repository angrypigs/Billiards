import os, sys
import pygame
from src.table_pockets_render import calculate_holes, POCKET_RADIUS
from math import sqrt
import numpy as np

WIDTH = 1700
HEIGHT = 1000
DIAMETER = int(sqrt(WIDTH**2 + HEIGHT**2))
FPS = 120

BALL_QUANTITY = 15

MAX_POWER = 20
RADIUS = 18
THRESHOLD = 5
LIMIT = THRESHOLD + RADIUS
CUE_RADIUS = 80
ERROR_THRESHOLD = 15

HOLES = calculate_holes(width=WIDTH, height=HEIGHT-100)

START_POS = (
    (0, 0),
    (1.75, -1),
    (1.75, 1),
    (3.5, -2),
    (3.5, 0),
    (3.5, 2),
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
    (100, 0, 20),
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

pygame.init()
pygame.font.init()

FONTS = {
    24: pygame.font.SysFont('Arial', 24),
    40: pygame.font.SysFont('Arial', 40)
}

temp = [f"{y}_{x}" for x in range(16) for y in "xy"]
temp.extend(["ball", "angle", "power", "score"])
COLUMN_NAMES = temp.copy()
del temp



def res_path(rel_path: str) -> str:
    """
    Return path to file modified by auto_py_to_exe path if packed to exe already
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = sys.path[0]
    return os.path.normpath(os.path.join(base_path, rel_path))

TRAINING_DATA_PATH_1PLAYER = res_path("assets/training_data/model_1player.db")
CHECKPOINT_PATH_1PLAYER = res_path("assets/models/model_1player.pt")



def is_point_in_rectangle_buffer(A: tuple[float, float], 
                                 B: tuple[float, float], 
                                 P: tuple[float, float], 
                                 R: float):
    x1, y1 = A
    x2, y2 = B
    x, y = P
    vx = x2 - x1
    vy = y2 - y1
    if vx == 0 and vy == 0:
        dist_sq = (x - x1)**2 + (y - y1)**2
        return dist_sq <= R**2
    t_A = vx * x1 + vy * y1
    t_B = vx * x2 + vy * y2
    t_P = vx * x + vy * y
    if not (min(t_A, t_B) <= t_P <= max(t_A, t_B)):
        return False
    len_AB_sq = vx*vx + vy*vy
    len_AB = sqrt(len_AB_sq)
    cross = abs(vx * (y - y1) - vy * (x - x1))
    distance = cross / len_AB
    return distance <= R

def line_hits_mask(mask: pygame.mask.Mask, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < mask.get_size()[0] and 0 <= y1 < mask.get_size()[1]:
            if mask.get_at((x1, y1)):
                return True

        if x1 == x2 and y1 == y2:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return False

def estimate_normal(mask: pygame.mask, cx: int, cy: int) -> tuple[int, int]:
    gx = mask.get_at((min(cx + 1, mask.get_size()[0] - 1), cy)) - mask.get_at((max(cx - 1, 0), cy))
    gy = mask.get_at((cx, min(cy + 1, mask.get_size()[1] - 1))) - mask.get_at((cx, max(cy - 1, 0)))
    length = (gx**2 + gy**2)**0.5
    if length == 0:
        return (0, -1)
    return (gx / length, gy / length)

def cosine_similarity(a: tuple[float, float], b: tuple[float, float]) -> float:
    dot = a[0] * b[0] + a[1] * b[1]
    norm_a = sqrt(a[0]**2 + a[1]**2)
    norm_b = sqrt(b[0]**2 + b[1]**2)
    return dot / (norm_a * norm_b)

def transform_to_relative(X):
    X_rel = X.copy()
    white_x = X[:, 0:1] 
    white_y = X[:, 1:2]
    num_balls = 15
    for i in range(num_balls):
        idx_x = 2 + (i * 2)
        idx_y = 3 + (i * 2)
        mask = (X[:, idx_x] != -1)
        X_rel[mask, idx_x] = X[mask, idx_x] - white_x[mask, 0]
        X_rel[mask, idx_y] = X[mask, idx_y] - white_y[mask, 0]
        X_rel[~mask, idx_x] = 0.0
        X_rel[~mask, idx_y] = 0.0
    return X_rel

def map_angle_to_agent_output(angle_deg: float, bounds: tuple[float, float]) -> float:
    min_a, max_a = bounds
    if abs(max_a - min_a) < 0.001:
        return 0.0

    clamped = max(min_a, min(angle_deg, max_a))

    ratio = (clamped - min_a) / (max_a - min_a)

    return 2.0 * ratio - 1.0

IMAGES = {}

LOAD_FLAG = [True]

def init_assets() -> None:
    if LOAD_FLAG[0]:
        LOAD_FLAG[0] = False
        IMAGES["table"] = pygame.image.load("assets/textures/table.png").convert_alpha()
        IMAGES["table_bg"] = pygame.image.load("assets/textures/table_bg.png").convert_alpha()
        IMAGES["cue"] = pygame.Surface((100, 100), pygame.SRCALPHA)
        pygame.draw.rect(IMAGES["cue"], (0, 0, 0), (0, 40, 100, 20))