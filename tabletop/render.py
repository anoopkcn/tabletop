#!/usr/bin/env python3
"""The tabletop dataset: one object on a table, seen by a colour camera and a
depth camera from the same viewpoint, rendered by a small ray caster
(perspective camera, Lambertian shading with ambient light, one cast
shadow, a near-white stripe and a knob on the object's front so that rotation
is visible on every shape in colour and in depth). One frame per state of the complete factor grid

    shape 4 x hue 12 x size 8 x rotation 12 x x-position 8 x light 4 x table 2
    = 294,912 states,

in row-major order of that grid (as 3dshapes.h5). Nuisance, seeded by the
state index and never captioned: camera pitch and yaw jitter (uniform, up to
JITTER_DEG), drawn independently for the colour camera and the depth camera
so that no state can be matched across the two by its jitter, and Gaussian
pixel noise on RGB (NOISE_STD in [0, 1] units).

Output (HDF5): images uint8 [N, 64, 64, 3]; depth uint16 [N, 64, 64] (ray
length in units of DEPTH_MAX / 65535); labels int8 [N, 7] (level index per
factor); values float32 [N, 7] (physical value per factor); jitter float32
[N, 4] (pitch and yaw of the colour camera, then of the depth camera, in
degrees). Attributes: factor names, counts, units.

    python -m tabletop --out data/tabletop.h5 --workers 12   # the grid
    python -m tabletop --out data/smoke.h5 --smoke           # 6,144 states
    python -m tabletop --out data/mini.h5 --mini             # 512 states
    python -m tabletop --preview preview.png                 # a sample sheet

Two of the three caption variants live here so that every consumer agrees:
caption_categorical (one token per level, colour and size words, named lights
and tables) and caption_ordinal (hue, size, rotation and position as run
lengths of one token each, level v written as v + 1 repeats). The third,
a fluent English sentence, is in captions_nl.py.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# the factor grid
FACTORS = ["shape", "hue", "size", "rotation", "x", "light", "table"]
COUNTS = (4, 12, 8, 12, 8, 4, 2)
N_STATES = int(np.prod(COUNTS))
SHAPES = ["cube", "sphere", "cylinder", "capsule"]
HUE_DEG = np.arange(12) * 30.0                       # periodic
SIZE_R = 0.20 + 0.026 * np.arange(8)                 # bounding half-height, scene units
ROT_DEG = np.arange(12) * 30.0                       # periodic, about the vertical
X_POS = np.linspace(-0.55, 0.55, 8)                  # along the table
LIGHT_DEG = 45.0 + 90.0 * np.arange(4)               # azimuth, periodic
LIGHT_ELEV = 45.0
TABLE_RGB = np.array([[0.74, 0.64, 0.48], [0.50, 0.58, 0.68]])   # beige, grey-blue
TABLE_NAMES = ["beige", "grey"]
LIGHT_NAMES = ["frontleft", "backleft", "backright", "frontright"]
HUE_NAMES = ["red", "orange", "yellow", "lime", "green", "teal", "cyan", "azure", "blue", "violet", "magenta", "rose"]
SIZE_NAMES = ["minute", "tiny", "small", "smallish", "medium", "large", "huge", "giant"]
VALUES = [np.arange(4.0), HUE_DEG, SIZE_R, ROT_DEG, X_POS, LIGHT_DEG, np.arange(2.0)]
UNITS = ["id", "degrees", "scene units", "degrees", "scene units", "degrees", "id"]

# camera, scene and nuisance
H = W = 64
SS = 2                                               # supersampling per axis
EYE = np.array([0.0, -2.3, 1.25])
TARGET = np.array([0.0, 0.0, 0.22])
VFOV_DEG = 40.0
TABLE_HALF = 2.0                                     # table is the square |x|, |y| <= TABLE_HALF at z = 0
WALL_Y = 2.2
WALL_RGB = np.array([0.36, 0.37, 0.43])
SKY_RGB = np.array([0.10, 0.11, 0.14])
AMBIENT, DIFFUSE, SHADOW = 0.30, 0.70, 0.8
STRIPE_AZ, STRIPE_HALF_DEG, STRIPE_WHITE = -90.0, 30.0, 0.8   # the front faces the camera at rotation 0; stripe colour = base mixed with white
KNOB_R = 0.30                                       # knob radius as a fraction of the size; it sits on the front at mid height
DEPTH_MAX = 8.0
JITTER_DEG = 1.0
NOISE_STD = 0.015


def state_levels(index):
    """Level indices [7] of a state index (row-major over COUNTS)."""
    return np.array(np.unravel_index(int(index), COUNTS), dtype=np.int64)


def state_index(levels):
    return int(np.ravel_multi_index(tuple(int(v) for v in levels), COUNTS))


def all_levels():
    return np.stack(np.unravel_index(np.arange(N_STATES), COUNTS), 1).astype(np.int8)


def physical(levels):
    return np.array([VALUES[k][int(v)] for k, v in enumerate(levels)], dtype=np.float32)


def hsv_to_rgb(h_deg, s=0.85, v=0.92):
    h = (h_deg % 360.0) / 60.0
    i = int(np.floor(h)) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])


# ----------------------------------------------------------------------------
# captions
def caption_categorical(levels):
    s, h, z, r, x, l, t = (int(v) for v in levels)
    return f"{SHAPES[s]} {HUE_NAMES[h]} {SIZE_NAMES[z]} r{r} x{x} {LIGHT_NAMES[l]} {TABLE_NAMES[t]}"


def caption_ordinal(levels):
    s, h, z, r, x, l, t = (int(v) for v in levels)
    return " ".join([SHAPES[s]] + ["h"] * (h + 1) + ["z"] * (z + 1) + ["r"] * (r + 1) + ["x"] * (x + 1) + [LIGHT_NAMES[l], TABLE_NAMES[t]])


CAPTIONS = {"categorical": caption_categorical, "ordinal": caption_ordinal}


# ----------------------------------------------------------------------------
# the ray caster (all pixels of one frame at once)
def _normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_rays(pitch_deg=0.0, yaw_deg=0.0, ss=SS):
    """Unit ray directions [H*ss, W*ss, 3] for the jittered camera."""
    f = _normalize(TARGET - EYE)
    f = _rot_z(yaw_deg) @ f
    right = _normalize(np.cross(f, np.array([0.0, 0.0, 1.0])))
    # pitch about the camera's right axis (Rodrigues)
    a = np.radians(pitch_deg)
    f = f * np.cos(a) + np.cross(right, f) * np.sin(a)
    up = np.cross(right, f)
    n = H * ss
    py = (np.arange(n) + 0.5) / n * 2 - 1
    px = (np.arange(W * ss) + 0.5) / (W * ss) * 2 - 1
    ty = np.tan(np.radians(VFOV_DEG / 2))
    d = f[None, None] + (px[None, :, None] * ty * (W / H)) * right[None, None] - (py[:, None, None] * ty) * up[None, None]
    return _normalize(d)


def _sphere(o, d, radius):
    b = np.einsum("ij,ij->i", o, d)
    c = np.einsum("ij,ij->i", o, o) - radius**2
    disc = b * b - c
    ok = disc > 0
    t = -b - np.sqrt(np.where(ok, disc, 0.0))
    t = np.where(ok & (t > 1e-6), t, np.inf)
    p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
    return t, _normalize(np.where(np.isfinite(t)[:, None], p, 1.0))


def _box(o, d, half):
    inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t1 = (-half - o) * inv
    t2 = (half - o) * inv
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    tnear, tfar = tmin.max(1), tmax.min(1)
    ok = (tnear < tfar) & (tfar > 1e-6) & (tnear > 1e-6)
    t = np.where(ok, tnear, np.inf)
    axis = tmin.argmax(1)
    n = np.zeros_like(o)
    n[np.arange(len(o)), axis] = -np.sign(d[np.arange(len(o)), axis])
    return t, n


def _cylinder_side(o, d, radius, hmin, hmax):
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = o[:, 0] * d[:, 0] + o[:, 1] * d[:, 1]
    c = o[:, 0] ** 2 + o[:, 1] ** 2 - radius**2
    disc = b * b - a * c
    ok = (disc > 0) & (a > 1e-12)
    t = (-b - np.sqrt(np.where(ok, disc, 0.0))) / np.where(a > 1e-12, a, 1.0)
    z = o[:, 2] + t * d[:, 2]
    ok &= (t > 1e-6) & (z >= hmin) & (z <= hmax)
    t = np.where(ok, t, np.inf)
    p = o + np.where(ok, t, 0.0)[:, None] * d
    n = np.stack([p[:, 0], p[:, 1], np.zeros_like(t)], 1) / radius
    return t, np.where(np.isfinite(t)[:, None], n, 0.0)


def _disc(o, d, radius, z0, sign):
    dz = np.where(np.abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
    t = (z0 - o[:, 2]) / dz
    p = o + np.where(np.isfinite(t), t, 0.0)[:, None] * d
    ok = (t > 1e-6) & (p[:, 0] ** 2 + p[:, 1] ** 2 <= radius**2)
    n = np.zeros_like(o)
    n[:, 2] = sign
    return np.where(ok, t, np.inf), n


def _merge(*hits):
    t = np.full(len(hits[0][0]), np.inf)
    n = np.zeros((len(t), 3))
    for ti, ni in hits:
        closer = ti < t
        t = np.where(closer, ti, t)
        n = np.where(closer[:, None], ni, n)
    return t, n


FRONT_RADIUS = {"sphere": 1.0, "cube": 1.0, "cylinder": 0.75, "capsule": 0.6}   # body radius on the front, as a fraction of the size


def knob_centre(shape, r):
    a = np.radians(STRIPE_AZ)
    return np.array([np.cos(a), np.sin(a), 0.0]) * FRONT_RADIUS[shape] * r


def intersect_body(o, d, shape, r):
    if shape == "sphere":
        return _sphere(o, d, r)
    if shape == "cube":
        return _box(o, d, r)
    if shape == "cylinder":
        rr = 0.75 * r
        return _merge(_cylinder_side(o, d, rr, -r, r), _disc(o, d, rr, r, 1.0), _disc(o, d, rr, -r, -1.0))
    if shape == "capsule":
        rr, hc = 0.6 * r, 0.4 * r
        top = _sphere(o - np.array([0.0, 0.0, hc]), d, rr)
        bot = _sphere(o + np.array([0.0, 0.0, hc]), d, rr)
        side = _cylinder_side(o, d, rr, -hc, hc)
        return _merge(side, top, bot)
    raise ValueError(shape)


def intersect_object(o, d, shape, r):
    """Ray-object intersection in the object frame (body plus the front knob):
    t (inf on a miss), unit normals, and whether the knob was hit."""
    tb, nb = intersect_body(o, d, shape, r)
    tk, nk = _sphere(o - knob_centre(shape, r), d, KNOB_R * r)
    knob = tk < tb
    return np.where(knob, tk, tb), np.where(knob[:, None], nk, nb), knob


def cast(levels, pitch, yaw, ss=SS):
    """One ray cast of the scene from the jittered camera: RGB float [H, W, 3]
    in [0, 1] without pixel noise, and depth float [H, W] (ray length)."""
    s, h, z, r, x, l, t = (int(v) for v in levels)
    shape, hue, rad, rot, xpos, laz, table = SHAPES[s], HUE_DEG[h], SIZE_R[z], ROT_DEG[r], X_POS[x], LIGHT_DEG[l], TABLE_RGB[t]
    dirs = camera_rays(pitch, yaw, ss).reshape(-1, 3)
    n_rays = len(dirs)
    eye = np.broadcast_to(EYE, dirs.shape)
    centre = np.array([xpos, 0.0, rad])
    R = _rot_z(rot)
    Rt = R.T
    le, la = np.radians(LIGHT_ELEV), np.radians(laz)
    light = np.array([np.cos(le) * np.cos(la), np.cos(le) * np.sin(la), np.sin(le)])
    base = hsv_to_rgb(hue)
    # object
    oo = (eye - centre) @ Rt.T
    dd = dirs @ Rt.T
    t_obj, n_loc, knob = intersect_object(oo, dd, shape, rad)
    p_loc = oo + np.where(np.isfinite(t_obj), t_obj, 0.0)[:, None] * dd
    az = np.degrees(np.arctan2(p_loc[:, 1], p_loc[:, 0]))
    stripe = ((np.abs((az - STRIPE_AZ + 180.0) % 360.0 - 180.0) < STRIPE_HALF_DEG) & (p_loc[:, 2] < 0.85 * rad)) | knob
    n_obj = n_loc @ R.T
    # table (finite square at z = 0)
    dz = np.where(np.abs(dirs[:, 2]) < 1e-12, 1e-12, dirs[:, 2])
    t_tab = -EYE[2] / dz
    p_tab = eye + t_tab[:, None] * dirs
    t_tab = np.where((t_tab > 1e-6) & (np.abs(p_tab[:, 0]) <= TABLE_HALF) & (np.abs(p_tab[:, 1]) <= TABLE_HALF), t_tab, np.inf)
    # back wall (vertical plane y = WALL_Y, above the table)
    dy = np.where(np.abs(dirs[:, 1]) < 1e-12, 1e-12, dirs[:, 1])
    t_wall = (WALL_Y - EYE[1]) / dy
    p_wall = eye + t_wall[:, None] * dirs
    t_wall = np.where((t_wall > 1e-6) & (p_wall[:, 2] >= 0.0), t_wall, np.inf)
    t_all = np.stack([t_obj, t_tab, t_wall], 1)
    which = t_all.argmin(1)
    t_hit = t_all[np.arange(n_rays), which]
    sky = ~np.isfinite(t_hit)
    depth = np.where(sky, DEPTH_MAX, np.minimum(t_hit, DEPTH_MAX))
    rgb = np.empty((n_rays, 3))
    m = (which == 0) & ~sky
    col = np.where(stripe[m][:, None], base * (1 - STRIPE_WHITE) + STRIPE_WHITE, base)
    ndl = np.clip(n_obj[m] @ light, 0.0, None)
    rgb[m] = col * (AMBIENT + DIFFUSE * ndl[:, None])
    m = (which == 1) & ~sky
    ndl = light[2]
    po = (p_tab[m] - centre) @ Rt.T
    ld = np.broadcast_to(light @ Rt.T, po.shape)
    t_sh, _, _ = intersect_object(po + 1e-4 * ld, ld, shape, rad)
    shadow = np.isfinite(t_sh)
    rgb[m] = table * (AMBIENT + DIFFUSE * ndl * np.where(shadow, 1.0 - SHADOW, 1.0))[:, None]
    m = (which == 2) & ~sky
    rgb[m] = WALL_RGB * (AMBIENT + DIFFUSE * max(0.0, -light[1]) * 0.5 + 0.35)
    rgb[sky] = SKY_RGB
    rgb = rgb.reshape(H, ss, W, ss, 3).mean((1, 3))
    depth = depth.reshape(H, ss, W, ss).mean((1, 3))
    return rgb, depth


def render(levels, seed=None, ss=SS, nuisance=True):
    """RGB float [H, W, 3] in [0, 1] from the colour camera, depth float
    [H, W] (ray length) from the depth camera, and the jitter draws
    [pitch_rgb, yaw_rgb, pitch_depth, yaw_depth] in degrees."""
    if seed is None:
        seed = state_index(levels)
    rng = np.random.default_rng(seed)
    if nuisance:
        j_rgb = rng.uniform(-JITTER_DEG, JITTER_DEG, 2)
        j_depth = rng.uniform(-JITTER_DEG, JITTER_DEG, 2)
    else:
        j_rgb = j_depth = np.zeros(2)
    rgb, _ = cast(levels, j_rgb[0], j_rgb[1], ss)
    depth = cast(levels, j_depth[0], j_depth[1], ss)[1] if nuisance else cast(levels, 0.0, 0.0, ss)[1]
    if nuisance and NOISE_STD > 0:
        rgb = rgb + rng.normal(0.0, NOISE_STD, rgb.shape)
    return np.clip(rgb, 0.0, 1.0), depth, np.concatenate([j_rgb, j_depth]).astype(np.float32)


def render_uint(levels, seed=None, nuisance=True):
    rgb, depth, jit = render(levels, seed, nuisance=nuisance)
    return (np.round(rgb * 255)).astype(np.uint8), np.round(depth / DEPTH_MAX * 65535).astype(np.uint16), jit


# ----------------------------------------------------------------------------
NUISANCE = [True]


def _chunk(args):
    """Render the states with indices rows[start:stop] (rows = state indices)."""
    start, rows, nuisance = args
    lv = np.stack(np.unravel_index(np.asarray(rows), COUNTS), 1)
    im = np.empty((len(rows), H, W, 3), np.uint8)
    dp = np.empty((len(rows), H, W), np.uint16)
    jt = np.empty((len(rows), 4), np.float32)
    for i, levels in enumerate(lv):
        im[i], dp[i], jt[i] = render_uint(levels, int(rows[i]), nuisance)
    return start, im, dp, jt


def subset_rows(**levels):
    """State indices of the sub-grid that keeps only the named levels of the
    named factors, complete on every factor not named."""
    lv = all_levels().astype(np.int64)
    keep = np.ones(len(lv), bool)
    for factor, values in levels.items():
        keep &= np.isin(lv[:, FACTORS.index(factor)], values)
    return np.flatnonzero(keep)


def smoke_rows():
    """A structured sub-grid that is complete on its own: hue in {0, 3, 6, 9},
    rotation in {0, 3, 6, 9}, x in {0, 3, 7}, light in {0, 2}: 6,144 states."""
    return subset_rows(hue=[0, 3, 6, 9], rotation=[0, 3, 6, 9], x=[0, 3, 7], light=[0, 2])


def mini_rows():
    """The smallest sub-grid that still touches every factor: 512 states,
    about 8 MB on disk, small enough to ship with the source."""
    return subset_rows(hue=[0, 6], size=[0, 6], rotation=[0, 3, 6, 9], x=[0, 7], light=[0, 2])


def write_grid(out: Path, workers: int, rows=None, chunk=1024, nuisance=True):
    from multiprocessing import Pool

    import h5py
    t0 = time.time()
    n = N_STATES if rows is None else len(rows)
    with h5py.File(out, "w") as f:
        f.create_dataset("images", (n, H, W, 3), np.uint8, chunks=(256, H, W, 3), compression="gzip", compression_opts=1)
        f.create_dataset("depth", (n, H, W), np.uint16, chunks=(256, H, W), compression="gzip", compression_opts=1)
        lv = all_levels() if rows is None else all_levels()[rows]
        f.create_dataset("labels", data=lv)
        f.create_dataset("values", data=np.stack([np.asarray(VALUES[k], np.float32)[lv[:, k].astype(int)] for k in range(7)], 1))
        f.create_dataset("state_index", data=np.arange(N_STATES) if rows is None else np.asarray(rows))
        f.create_dataset("jitter", (n, 4), np.float32)
        f.attrs["factors"] = FACTORS
        f.attrs["counts"] = COUNTS
        f.attrs["units"] = UNITS
        f.attrs["depth_max"] = DEPTH_MAX
        f.attrs["jitter_deg"] = JITTER_DEG if nuisance else 0.0
        f.attrs["noise_std"] = NOISE_STD if nuisance else 0.0
        f.attrs["nuisance"] = bool(nuisance)
        f.attrs["shapes"] = SHAPES
        idx = np.arange(N_STATES) if rows is None else np.asarray(rows)
        tasks = [(st, idx[st:st + chunk], nuisance) for st in range(0, n, chunk)]
        done = 0
        with Pool(workers) as pool:
            for k, (start, im, dp, jt) in enumerate(pool.imap_unordered(_chunk, tasks)):
                f["images"][start:start + len(im)] = im
                f["depth"][start:start + len(im)] = dp
                f["jitter"][start:start + len(im)] = jt
                done += len(im)
                if k % 20 == 0:
                    print(f"  {done} / {n} frames ({time.time() - t0:.0f} s)", flush=True)
    print(f"wrote {out} with {n} states in {time.time() - t0:.0f} s")


def preview(path: Path):
    """A sheet: rows vary one factor each from a base state (cube, red, medium, front, centre, front-left, beige)."""
    from PIL import Image
    base = np.array([0, 0, 4, 0, 3, 0, 0])
    rows = []
    for k, name in enumerate(FACTORS):
        vals = list(range(COUNTS[k])) if COUNTS[k] <= 8 else list(range(0, COUNTS[k], COUNTS[k] // 8))[:8]
        tiles = []
        for v in vals:
            lv = base.copy(); lv[k] = v
            rgb, depth, _ = render(lv)
            d = np.clip((depth - 1.6) / 3.2, 0, 1)
            tiles.append(np.concatenate([rgb, np.repeat(d[:, :, None], 3, 2)], 0))
        while len(tiles) < 8:
            tiles.append(np.zeros_like(tiles[0]))
        rows.append(np.concatenate(tiles, 1))
    sheet = (np.concatenate(rows, 0) * 255).astype(np.uint8)
    Image.fromarray(sheet).resize((sheet.shape[1] * 2, sheet.shape[0] * 2), Image.NEAREST).save(path)
    print(f"wrote {path}: one row per factor {FACTORS}, RGB above depth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="the 6,144-state sub-grid")
    p.add_argument("--mini", action="store_true", help="the 512-state sub-grid shipped with the source")
    p.add_argument("--preview", type=Path)
    p.add_argument("--time", action="store_true", help="time 50 renders")
    p.add_argument("--no-nuisance", action="store_true", help="no camera jitter and no pixel noise (the residual control)")
    a = p.parse_args()
    if a.preview:
        preview(a.preview)
    if a.time:
        t0 = time.time()
        for i in range(50):
            render_uint(state_levels((i * 9973) % N_STATES), i)
        print(f"{(time.time() - t0) / 50 * 1000:.1f} ms per frame")
    if a.out:
        rows = mini_rows() if a.mini else smoke_rows() if a.smoke else None
        write_grid(a.out, a.workers, rows=rows, nuisance=not a.no_nuisance)


if __name__ == "__main__":
    main()
