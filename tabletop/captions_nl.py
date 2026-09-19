"""Natural-language captions of tabletop states, for pretrained text encoders.
Every factor is described in words and all 294,912 captions are unique;
rotation and position are the hard parts for an encoder that never learned
this scene.

    >>> caption([0, 3, 4, 6, 2, 1, 0])
    'a medium lime green cube, rotated 180 degrees clockwise, slightly left of center, lit from the back left, on a beige table'
"""

from __future__ import annotations

SHAPES = ["cube", "sphere", "cylinder", "capsule"]
HUES = ["red", "orange", "yellow", "lime green", "green", "teal", "cyan", "sky blue", "blue", "violet", "magenta", "pink"]
SIZES = ["very tiny", "tiny", "small", "smallish", "medium", "large", "very large", "huge"]
XS = ["far left", "left", "slightly left of center", "just left of center", "just right of center", "slightly right of center", "right", "far right"]
LIGHTS = ["front left", "back left", "back right", "front right"]
TABLES = ["beige", "grey"]


def caption(levels) -> str:
    s, h, z, r, x, l, t = (int(v) for v in levels)
    rot = f"rotated {30 * r} degrees clockwise" if r else "facing the camera"
    return f"a {SIZES[z]} {HUES[h]} {SHAPES[s]}, {rot}, {XS[x]}, lit from the {LIGHTS[l]}, on a {TABLES[t]} table"


if __name__ == "__main__":
    from tabletop.render import all_levels
    lv = all_levels()
    caps = [caption(v) for v in lv[:: 4801]]
    print("\n".join(caps[:6]))
    print(len({caption(v) for v in lv}), "unique captions of", len(lv))
