"""
Assembles Fig. "attention decomposition" from the five per-term attention-weight
renders (source: Progress Review deck, slide 14 / notesSlide "Model Attention").
Panels 1-4 ship concatenated in one source frame; panel 5 ships standalone.
Run once from this directory: python3 mkattn.py
"""
from PIL import Image, ImageDraw, ImageFont

SRC = "figures/_attn_source"
OUT = "figures"

strip = Image.open(f"{SRC}/panels1-4.png").convert("RGB")
panel5 = Image.open(f"{SRC}/panel5.png").convert("RGB")

W, H = strip.size
pw = W // 4
panels = [strip.crop((i * pw, 0, (i + 1) * pw, H)) for i in range(4)]

# match panel 5 to the row height, preserving aspect ratio
scale = H / panel5.size[1]
panel5 = panel5.resize((int(panel5.size[0] * scale), H), Image.LANCZOS)

GAP = 10
BORDER = "#d8d6d0"
row_w = pw * 4 + GAP * 3
total_w = row_w + GAP * 2 + panel5.size[0]
total_h = H

canvas = Image.new("RGB", (total_w, total_h), "white")
x = 0
for p in panels:
    canvas.paste(p, (x, 0))
    x += pw + GAP
canvas.paste(panel5, (row_w + GAP * 2, 0))

draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
except OSError:
    font = ImageFont.load_default()

xs = [0, pw + GAP, 2 * (pw + GAP), 3 * (pw + GAP), row_w + GAP * 2]
for i, lx in enumerate(xs, start=1):
    draw.rectangle([lx + 6, 6, lx + 40, 42], fill="black")
    draw.text((lx + 14, 8), str(i), fill="white", font=font)

canvas.save(f"{OUT}/attention.png")
canvas.save(f"{OUT}/attention.pdf")
print("ok", canvas.size)
