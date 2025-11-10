from PIL import Image, ImageDraw
from math import sqrt

img = Image.open("table_src.png").convert("RGB")
draw = ImageDraw.Draw(img)

width, height = img.size
pocket_radius = 20
offset = 28
offset_mid = 43

field_color = img.getpixel((width // 2, height // 2))

holes = [
    (offset_mid, offset_mid),
    (width // 2, offset),
    (width - offset_mid, offset_mid),
    (offset_mid, height - offset_mid),
    (width // 2, height - offset),
    (width - offset_mid, height - offset_mid),
]

pockets = [
    [(1, -1), (-1, 1), (1, 1), (1, 1)],
    [(-1, 0), (1, 0), (-1, 1), (1, 1)],
    [(-1, -1), (1, 1), (-1, 1), (-1, 1)],
    [(-1, -1), (1, 1), (1, -1), (1, -1)],
    [(-1, 0), (1, 0), (-1, -1), (1, -1)],
    [(-1, 1), (1, -1), (-1, -1), (-1, -1)]
]
for i in (0, 2, 3, 5):
    for j in range(4):
        pockets[i][j] = tuple([z * sqrt(2) / 2 for z in pockets[i][j]])

for i, (x, y) in enumerate(holes):
    pw, ph, aw, ah = pockets[i]
    points = [(x + pw[0] * pocket_radius, y + pw[1] * pocket_radius), 
              (x + ph[0] * pocket_radius, y + ph[1] * pocket_radius)]
    points.append((points[1][0] + ah[0] * pocket_radius * 2, points[1][1] + ah[1] * pocket_radius * 2))
    points.append((points[0][0] + aw[0] * pocket_radius * 2, points[0][1] + aw[1] * pocket_radius * 2))
    draw.polygon(points, fill=field_color)
    draw.ellipse((x - pocket_radius, y - pocket_radius, x + pocket_radius, y + pocket_radius),
                 fill="black")

img = img.convert("RGBA")
datas = img.getdata()

new_data = []
for item in datas:
    if item[:3] == field_color[:3]:
        new_data.append((0, 0, 0, 0))
    else:
        new_data.append(item)

img.putdata(new_data)

img.show()
img.save("table.png")