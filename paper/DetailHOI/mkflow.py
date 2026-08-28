"""
Fig. 1: the pipeline flow chart, taken directly from the Progress Review deck
(slide 10, "Flow Chart"), not redrawn. Source is a 300dpi render of that slide
kept at figures/_flow_source/slide10_raw.png. This script only crops the
slide's title/page-number/university-logo chrome and paints out a leftover
corner of the slide template's decorative triangle (it sits in blank
background, clear of every label) -- the diagram, arrows, and both photos are
untouched pixels from the original slide.

Note: the input/output photos in this figure carry a "Mustafa Photography"
watermark (bottom-right corner of each photo) -- that credit is not resolved;
see README.md.

Run once from this directory: python3 mkflow.py
"""
from PIL import Image, ImageDraw

SRC = "figures/_flow_source/slide10_raw.png"
OUT = "figures"

im = Image.open(SRC).convert("RGB")
draw = ImageDraw.Draw(im)
draw.rectangle([3350, 0, im.width, 790], fill="white")  # leftover template triangle
crop = im.crop((0, 260, 3920, 2040))

crop.save(f"{OUT}/flowchart.png")
crop.save(f"{OUT}/flowchart.pdf")
print("ok", crop.size)
