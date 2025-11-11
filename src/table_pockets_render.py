from PIL import Image, ImageDraw
from math import sqrt

POCKET_RADIUS = 20
OFFSET = 28
OFFSET_MID = 43

def calculate_holes(width: int, height: int):
    return [
        (OFFSET_MID, OFFSET_MID),
        (width // 2, OFFSET),
        (width - OFFSET_MID, OFFSET_MID),
        (OFFSET_MID, height - OFFSET_MID),
        (width // 2, height - OFFSET),
        (width - OFFSET_MID, height - OFFSET_MID),
    ]

POCKET_SHAPES = [
    [(1, -1), (-1, 1), (1, 1), (1, 1)],
    [(-1, 0), (1, 0), (-1, 1), (1, 1)],
    [(-1, -1), (1, 1), (-1, 1), (-1, 1)],
    [(-1, -1), (1, 1), (1, -1), (1, -1)],
    [(-1, 0), (1, 0), (-1, -1), (1, -1)],
    [(-1, 1), (1, -1), (-1, -1), (-1, -1)]
]

for i in (0, 2, 3, 5):
    for j in range(4):
        POCKET_SHAPES[i][j] = tuple([z * sqrt(2) / 2 for z in POCKET_SHAPES[i][j]])

def render_pockets(
    src_path: str = "assets/textures/table_src.png",
    dst_path: str = "assets/textures/table.png",
    with_holes: bool = False) -> Image.Image:
    img = Image.open(src_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    field_color = img.getpixel((width // 2, height // 2))
    HOLES = calculate_holes(width, height)

    for i, (x, y) in enumerate(HOLES):
        pw, ph, aw, ah = POCKET_SHAPES[i]
        points = [
            (x + pw[0] * POCKET_RADIUS, y + pw[1] * POCKET_RADIUS),
            (x + ph[0] * POCKET_RADIUS, y + ph[1] * POCKET_RADIUS)
        ]
        points.append((points[1][0] + ah[0] * POCKET_RADIUS * 2, points[1][1] + ah[1] * POCKET_RADIUS * 2))
        points.append((points[0][0] + aw[0] * POCKET_RADIUS * 2, points[0][1] + aw[1] * POCKET_RADIUS * 2))
        draw.polygon(points, fill=field_color)

        if with_holes:
            draw.ellipse(
                (x - POCKET_RADIUS, y - POCKET_RADIUS, x + POCKET_RADIUS, y + POCKET_RADIUS),
                fill=(1, 1, 1)
            )

    img = img.convert("RGBA")
    datas = img.getdata()
    new_data = [
        (0, 0, 0, 0) if item[:3] == field_color[:3] else item
        for item in datas
    ]
    img.putdata(new_data)

    img.show()
    img.save(dst_path)

    return img

__all__ = ["POCKET_RADIUS", "POCKET_SHAPES", "calculate_holes", "render_pockets"]