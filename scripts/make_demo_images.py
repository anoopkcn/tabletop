#!/usr/bin/env python3
"""Draw the figures the README shows, straight from the renderer (no data file
needed). Every panel is a real render of a real state, upscaled with nearest
neighbour so that a 64 x 64 frame stays readable.

    python scripts/make_demo_images.py                  # all of them, into assets/
    python scripts/make_demo_images.py --only hero rotation
    python scripts/make_demo_images.py --out /tmp/fig --scale 2

Needs numpy and pillow only; the depth colour map is built in. Each sheet
carries its own dark background, so the figures read the same on a light or a
dark page, and type is sized as a fraction of the sheet's width so that a wide
sheet does not end up with unreadable labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabletop.captions_nl import caption as caption_natural
from tabletop.render import (
    COUNTS,
    FACTORS,
    HUE_DEG,
    HUE_NAMES,
    LIGHT_NAMES,
    ROT_DEG,
    SHAPES,
    SIZE_NAMES,
    SIZE_R,
    TABLE_NAMES,
    X_POS,
    caption_categorical,
    render,
)

BG = (18, 20, 25)
INK = (232, 234, 240)
DIM = (140, 146, 162)
RULE = (52, 56, 68)
TILE = 64

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
BOLD_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]


def font(size: int, bold: bool = False):
    for path in BOLD_CANDIDATES if bold else FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# depth colouring: a five-stop ramp, near = warm, far = dark, over the range the
# scene actually spans (the near edge of the table out to the back wall)
DEPTH_STOPS = np.array(
    [[0.99, 0.95, 0.75], [0.96, 0.72, 0.36], [0.75, 0.42, 0.42], [0.36, 0.26, 0.50], [0.09, 0.10, 0.17]]
)
DEPTH_NEAR, DEPTH_FAR = 1.78, 5.0
DEPTH_GAMMA = 0.65                    # < 1 spreads the near end, where the object is


def depth_rgb(depth: np.ndarray) -> np.ndarray:
    """Depth in scene units -> RGB float [H, W, 3] in [0, 1]."""
    t = np.clip((depth - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR), 0.0, 1.0) ** DEPTH_GAMMA
    pos = t * (len(DEPTH_STOPS) - 1)
    i = np.clip(pos.astype(int), 0, len(DEPTH_STOPS) - 2)
    f = (pos - i)[..., None]
    return DEPTH_STOPS[i] * (1 - f) + DEPTH_STOPS[i + 1] * f


def frame(levels, kind: str = "rgb", scale: int = 3, nuisance: bool = True) -> Image.Image:
    rgb, depth, _ = render(np.asarray(levels), nuisance=nuisance)
    arr = rgb if kind == "rgb" else depth_rgb(depth)
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    return im.resize((TILE * scale, TILE * scale), Image.NEAREST)


# ----------------------------------------------------------------------------
class Sheet:
    """One figure. Type sizes are a fraction of the width, so a 2,500 px sheet
    and a 1,000 px sheet carry labels of the same apparent weight."""

    def __init__(self, width: int, height: int):
        self.im = Image.new("RGB", (width, height), BG)
        self.d = ImageDraw.Draw(self.im)
        self.w, self.h = width, height
        self.title_pt = max(20, round(width * 0.0145))
        self.sub_pt = max(13, round(width * 0.0090))
        self.label_pt = max(14, round(width * 0.0082))
        self.small_pt = max(11, round(width * 0.0062))

    def text(self, xy, s, pt, colour=INK, bold=False, anchor="la"):
        self.d.text(xy, s, font=font(pt, bold), fill=colour, anchor=anchor)

    def header(self, x, y, title, subtitle):
        self.text((x, y), title, self.title_pt, INK, bold=True)
        self.text((x, y + round(self.title_pt * 1.35)), subtitle, self.sub_pt, DIM)

    def paste(self, im, xy):
        self.im.paste(im, xy)

    def rule(self, y, x0, x1):
        self.d.line([(x0, y), (x1, y)], fill=RULE)

    def wrap(self, s, x, y, width, pt, colour, lines=2):
        f = font(pt)
        out, line = [], ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if self.d.textlength(trial, font=f) <= width or not line:
                line = trial
            else:
                out.append(line)
                line = word
        out.append(line)
        for i, ln in enumerate(out[:lines]):
            self.d.text((x, y + i * (pt + 3)), ln, font=f, fill=colour)

    def save(self, out: Path, name: str):
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.png"
        self.im.save(path, optimize=True)
        print(f"  {path.name:<16} {self.w} x {self.h}  ({path.stat().st_size / 1024:.0f} KB)")


def head_h(width: int) -> int:
    """Height reserved above the first row of frames."""
    return max(20, round(width * 0.0145)) * 3


# ----------------------------------------------------------------------------
# the figures
BASE = np.array([0, 0, 4, 0, 3, 0, 0])      # cube, red, medium, facing front, centre, front-left, beige


def fig_hero(out: Path, scale: int):
    """A wall of states, one shape per row, sampled far apart in the grid."""
    cols, rows = 12, 4
    picks = [
        [s, (7 * i) % 12, (3 + 5 * i) % 8, (2 * i + 3 * s) % 12, (i * 3 + s) % 8, (i + s) % 4, (i + s) % 2]
        for s in range(4)
        for i in range(cols)
    ]
    tw, gap = TILE * scale, 4
    pad = round(tw * 0.28)
    w = pad * 2 + cols * tw + (cols - 1) * gap
    top = head_h(w)
    sh = Sheet(w, pad * 2 + top + rows * tw + (rows - 1) * gap)
    sh.header(pad, pad, "tabletop", "294,912 states  ·  colour, depth and captions  ·  7 factors, complete grid")
    for k, levels in enumerate(picks):
        r, c = divmod(k, cols)
        sh.paste(frame(levels, "rgb", scale), (pad + c * (tw + gap), pad + top + r * (tw + gap)))
    sh.save(out, "hero")


def fig_factors(out: Path, scale: int):
    """One row per factor, sweeping that factor from the base state: RGB above depth."""
    sweeps = [(FACTORS[k], k, list(range(COUNTS[k]))) for k in range(len(FACTORS))]
    cols = max(COUNTS)
    tw, gap = TILE * scale, 4
    pad = round(tw * 0.28)
    gutter = round(tw * 1.0)
    w = pad * 2 + gutter + cols * (tw + gap)
    sh = Sheet(w, 0)
    rowh = tw * 2 + gap + sh.small_pt * 2 + 14
    sh = Sheet(w, pad * 2 + head_h(w) + len(sweeps) * rowh)
    sh.header(pad, pad, "one factor at a time", "every other factor held at the base state · colour above depth")
    y = pad + head_h(w)
    for name, k, vals in sweeps:
        sh.text((pad, y + tw - sh.label_pt), name, sh.label_pt, INK, bold=True)
        sh.text((pad, y + tw + 6), f"{COUNTS[k]} levels", sh.small_pt, DIM)
        for j, v in enumerate(vals):
            lv = BASE.copy()
            lv[k] = v
            x = pad + gutter + j * (tw + gap)
            sh.paste(frame(lv, "rgb", scale), (x, y))
            sh.paste(frame(lv, "depth", scale), (x, y + tw + gap))
            sh.text((x + tw // 2, y + tw * 2 + gap + 6), _level_label(k, v), sh.small_pt, DIM, anchor="ma")
        sh.rule(y + rowh - 8, pad, w - pad)
        y += rowh
    sh.save(out, "factors")


def _level_label(k: int, v: int) -> str:
    name = FACTORS[k]
    if name == "shape":
        return SHAPES[v]
    if name == "hue":
        return f"{HUE_NAMES[v]} · {HUE_DEG[v]:.0f}°"
    if name == "size":
        return f"{SIZE_NAMES[v]} · {SIZE_R[v]:.2f}"
    if name == "rotation":
        return f"r{v} · {ROT_DEG[v]:.0f}°"
    if name == "x":
        return f"x{v} · {X_POS[v]:+.2f}"
    if name == "light":
        return LIGHT_NAMES[v]
    return TABLE_NAMES[v]


def fig_modalities(out: Path, scale: int):
    """Six states as the three modalities see them: colour, depth, words."""
    picks = [
        [0, 0, 5, 0, 1, 0, 0],
        [1, 4, 3, 5, 6, 2, 1],
        [2, 8, 6, 9, 2, 1, 0],
        [3, 10, 2, 3, 5, 3, 1],
        [1, 6, 4, 7, 4, 0, 0],
        [2, 2, 4, 11, 7, 2, 1],
    ]
    tw, gap = TILE * scale, 4
    pad = round(tw * 0.28)
    gutter = round(tw * 1.0)
    w = pad * 2 + gutter + len(picks) * (tw + gap)
    sh = Sheet(w, 0)
    caph = sh.small_pt * 8 + 24
    sh = Sheet(w, pad * 2 + head_h(w) + tw * 2 + gap + caph)
    sh.header(pad, pad, "three views of one state", "the colour camera, the depth camera beside it, and the words")
    y = pad + head_h(w)
    sh.text((pad, y + tw // 2), "RGB", sh.label_pt, INK, bold=True, anchor="lm")
    sh.text((pad, y + tw + gap + tw // 2), "depth", sh.label_pt, INK, bold=True, anchor="lm")
    cap_y = y + tw * 2 + gap + 14
    sh.text((pad, cap_y), "caption", sh.label_pt, INK, bold=True)
    sh.text((pad, cap_y + sh.small_pt * 2), "categorical", sh.small_pt, DIM)
    sh.text((pad, cap_y + sh.small_pt * 5), "natural", sh.small_pt, DIM)
    for j, levels in enumerate(picks):
        x = pad + gutter + j * (tw + gap)
        sh.paste(frame(levels, "rgb", scale), (x, y))
        sh.paste(frame(levels, "depth", scale), (x, y + tw + gap))
        sh.wrap(caption_categorical(levels), x, cap_y + sh.small_pt * 2, tw - 6, sh.small_pt, INK, lines=2)
        sh.wrap(caption_natural(levels), x, cap_y + sh.small_pt * 5, tw - 6, sh.small_pt, DIM, lines=5)
    sh.save(out, "modalities")


def fig_rotation(out: Path, scale: int):
    """All twelve rotations of all four shapes, in colour and in depth: the
    stripe and the knob are what carry the turn into a sphere's depth map."""
    tw, gap = TILE * scale, 3
    pad = round(tw * 0.28)
    gutter = round(tw * 1.0)
    cols = 12
    w = pad * 2 + gutter + cols * (tw + gap)
    sh = Sheet(w, 0)
    band = tw * 2 + gap + round(sh.small_pt * 1.8)
    sh = Sheet(w, pad * 2 + head_h(w) + 4 * band + sh.small_pt)
    sh.header(
        pad,
        pad,
        "rotation is visible on every shape",
        "12 steps of 30° about the vertical · a near-white stripe and a front knob carry the turn into depth, "
        "where a sphere would otherwise be a featureless ball",
    )
    y = pad + head_h(w)
    for s in range(4):
        sh.text((pad, y + tw // 2), SHAPES[s], sh.label_pt, INK, bold=True, anchor="lm")
        sh.text((pad, y + tw + gap + tw // 2), "depth", sh.small_pt, DIM, anchor="lm")
        for r in range(cols):
            lv = BASE.copy()
            lv[0], lv[1], lv[2], lv[3] = s, (3 * s) % 12, 4, r
            x = pad + gutter + r * (tw + gap)
            sh.paste(frame(lv, "rgb", scale), (x, y))
            sh.paste(frame(lv, "depth", scale), (x, y + tw + gap))
            if s == 3:
                sh.text((x + tw // 2, y + tw * 2 + gap + 6), f"{30 * r}°", sh.small_pt, DIM, anchor="ma")
        y += band
    sh.save(out, "rotation")


def fig_hue_depth(out: Path, scale: int):
    """The twelve hues in colour and the same twelve states in depth: hue is not
    in the depth image at all, which bounds what a depth encoder can resolve."""
    tw, gap = TILE * scale, 3
    pad = round(tw * 0.28)
    gutter = round(tw * 1.0)
    w = pad * 2 + gutter + 12 * (tw + gap)
    sh = Sheet(w, 0)
    sh = Sheet(w, pad * 2 + head_h(w) + tw * 2 + gap + round(sh.small_pt * 2.2))
    sh.header(
        pad,
        pad,
        "what depth cannot see",
        "twelve hues of one shape · the colour row runs the wheel, the depth row only wobbles with the camera jitter",
    )
    y = pad + head_h(w)
    sh.text((pad, y + tw // 2), "RGB", sh.label_pt, INK, bold=True, anchor="lm")
    sh.text((pad, y + tw + gap + tw // 2), "depth", sh.label_pt, INK, bold=True, anchor="lm")
    for hue in range(12):
        lv = BASE.copy()
        lv[0], lv[1], lv[2] = 1, hue, 6
        x = pad + gutter + hue * (tw + gap)
        sh.paste(frame(lv, "rgb", scale), (x, y))
        sh.paste(frame(lv, "depth", scale), (x, y + tw + gap))
        sh.text((x + tw // 2, y + tw * 2 + gap + 6), HUE_NAMES[hue], sh.small_pt, DIM, anchor="ma")
    sh.save(out, "hue_depth")


def fig_nuisance(out: Path, scale: int):
    """The residual has a known cause and a known size: camera jitter drawn
    independently per camera, and Gaussian pixel noise on RGB."""
    picks = [[0, 0, 5, 2, 3, 0, 0], [1, 5, 4, 7, 2, 1, 1], [2, 9, 6, 4, 5, 2, 0], [3, 2, 3, 10, 6, 3, 1]]
    tw, gap = TILE * scale, 6
    pad = round(tw * 0.28)
    gutter = round(tw * 1.15)
    w = pad * 2 + gutter + len(picks) * (tw + gap)
    sh = Sheet(w, pad * 2 + head_h(w) + 3 * (tw + gap))
    sh.header(
        pad,
        pad,
        "the nuisance, and its size",
        "camera pitch and yaw within 1°, drawn independently per camera, and σ = 0.015 pixel noise on RGB",
    )
    y = pad + head_h(w)
    for i, (label, sub) in enumerate([("clean", "no nuisance"), ("as shipped", "jitter + noise"), ("difference", "×8, absolute")]):
        yy = y + i * (tw + gap) + tw // 2
        sh.text((pad, yy - sh.small_pt), label, sh.label_pt, INK, bold=True, anchor="lm")
        sh.text((pad, yy + sh.small_pt), sub, sh.small_pt, DIM, anchor="lm")
    for j, levels in enumerate(picks):
        lv = np.asarray(levels)
        clean, _, _ = render(lv, nuisance=False)
        dirty, _, _ = render(lv, nuisance=True)
        diff = np.clip(np.abs(dirty - clean) * 8.0, 0, 1)
        x = pad + gutter + j * (tw + gap)
        for i, arr in enumerate([clean, dirty, diff]):
            im = Image.fromarray((arr * 255).astype(np.uint8)).resize((tw, tw), Image.NEAREST)
            sh.paste(im, (x, y + i * (tw + gap)))
    sh.save(out, "nuisance")


FIGURES = {
    "hero": fig_hero,
    "factors": fig_factors,
    "modalities": fig_modalities,
    "rotation": fig_rotation,
    "hue_depth": fig_hue_depth,
    "nuisance": fig_nuisance,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "assets")
    p.add_argument("--scale", type=int, default=3, help="upscale factor per 64 x 64 frame")
    p.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="draw only these")
    a = p.parse_args()
    names = a.only or list(FIGURES)
    print(f"drawing {len(names)} figures into {a.out}")
    for name in names:
        FIGURES[name](a.out, a.scale)


if __name__ == "__main__":
    main()
