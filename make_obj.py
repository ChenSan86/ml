#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Square base + smooth global undulation + 3 bumps with different foot heights (pedestals) + watertight solid.
- Base: square slab (bottom at base_z, base top plane at base_z + base_thickness)
- Global undulation: low-frequency smooth wavy field, fades to 0 near boundary (optional)
- Each bump i:
    height += pedestal_i * w_i(r)  +  amp_i * gaussian(r) * w_i(r)
  where w_i is C2 quintic smooth cutoff (goes to 0 with zero slope at r = R_i)
  => smooth junction to surrounding surface (no crease).
- Watertight mesh: top surface + bottom surface + 4 side walls (outer square boundary)

Dependencies: numpy only

Example:
  python solid_smooth_heightfield_v2.py --n 256 --with_normals --with_uv --out_obj solid.obj --out_ply solid.ply
"""

from __future__ import annotations
import argparse
import numpy as np
from typing import Tuple


# ----------------------------
# Smooth helpers
# ----------------------------
def clamp01(t: np.ndarray) -> np.ndarray:
    return np.clip(t, 0.0, 1.0)


def smoothstep_quintic(t: np.ndarray) -> np.ndarray:
    """
    Quintic smoothstep: 6t^5 - 15t^4 + 10t^3
    C2 continuous; slope and curvature go to 0 at endpoints.
    """
    t = clamp01(t)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def gaussian(r2: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-r2 / (2.0 * sigma * sigma))


def c2_cutoff_weight(r: np.ndarray, R: float) -> np.ndarray:
    """
    Weight w(r) = smoothstep_quintic(1 - r/R)
    w(0)=1, w(R)=0, and dw/dr at r=R is 0 (smooth attachment).
    """
    t = 1.0 - (r / max(R, 1e-12))
    return smoothstep_quintic(t)


def boundary_fade_mask(
    x: np.ndarray, y: np.ndarray,
    x_min: float, x_max: float, y_min: float, y_max: float,
    fade_width: float
) -> np.ndarray:
    """
    Mask goes to 0 at the outer square boundary and becomes 1 inside after fade_width.
    Useful to enforce perfectly-flat top near edges (clean base boundary).
    """
    if fade_width <= 0.0:
        return np.ones_like(x, dtype=np.float64)

    d_left = x - x_min
    d_right = x_max - x
    d_bottom = y - y_min
    d_top = y_max - y
    d = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))

    t = d / max(fade_width, 1e-12)
    return smoothstep_quintic(t)


# ----------------------------
# Height field components
# ----------------------------
def global_undulation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Low-frequency smooth terrain-like undulation (peaks之间起伏).
    Keep amplitude small vs bumps.
    """
    # frequencies chosen to be low; combine a couple modes
    u = (
        0.60 * np.sin(2.0 * np.pi * (0.90 * x + 0.20 * y)) +
        0.45 * np.sin(2.0 * np.pi * (0.25 * x + 0.85 * y + 0.15)) +
        0.30 * np.cos(2.0 * np.pi * (0.65 * x - 0.30 * y))
    )
    # normalize-ish to [-1,1] range (roughly)
    return u / 1.35


def bumps_with_pedestals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sum of 3 convex bumps, each with:
      pedestal_i * w_i(r)  +  amp_i * gaussian(r) * w_i(r)
    => Different "foot heights" and smooth blending at r=R_i.
    """
    # (cx, cy, pedestal, amp, sigma, R_cutoff)
    bumps = [
        (0.44, 0.77, 0.30, 1.30, 0.16, 0.46),  # big peak 1, higher foot
        (0.28, 0.30, 0.12, 1.10, 0.18, 0.52),  # big peak 2, lower foot
        (0.84, 0.52, 0.22, 0.95, 0.09, 0.30),  # smaller peak, medium foot
    ]

    h = np.zeros_like(x, dtype=np.float64)

    for cx, cy, ped, amp, sigma, R in bumps:
        dx = x - cx
        dy = y - cy
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2)
        w = c2_cutoff_weight(r, R)

        # pedestal makes "mountain foot" higher/lower, still smooth to 0 at r=R
        h += ped * w

        # bump core
        h += amp * gaussian(r2, sigma) * w

    return h


# ----------------------------
# Mesh building
# ----------------------------
def build_grid(n: int, x_min: float, x_max: float, y_min: float, y_max: float) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, n, dtype=np.float64)
    ys = np.linspace(y_min, y_max, n, dtype=np.float64)
    return np.meshgrid(xs, ys, indexing="xy")


def triangulate_grid(n: int, invert: bool = False) -> np.ndarray:
    faces = []
    for j in range(n - 1):
        for i in range(n - 1):
            v00 = j * n + i
            v10 = j * n + (i + 1)
            v01 = (j + 1) * n + i
            v11 = (j + 1) * n + (i + 1)
            if not invert:
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))
            else:
                faces.append((v00, v11, v10))
                faces.append((v00, v01, v11))
    return np.array(faces, dtype=np.int64)


def boundary_loop_indices(n: int) -> np.ndarray:
    idx = []
    for i in range(n):                # top edge
        idx.append(0 * n + i)
    for j in range(1, n):             # right edge
        idx.append(j * n + (n - 1))
    for i in range(n - 2, -1, -1):    # bottom edge
        idx.append((n - 1) * n + i)
    for j in range(n - 2, 0, -1):     # left edge
        idx.append(j * n + 0)
    return np.array(idx, dtype=np.int64)


def compute_vertex_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    V = v.shape[0]
    nrm = np.zeros((V, 3), dtype=np.float64)
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    np.add.at(nrm, f[:, 0], fn)
    np.add.at(nrm, f[:, 1], fn)
    np.add.at(nrm, f[:, 2], fn)
    lens = np.linalg.norm(nrm, axis=1)
    lens = np.maximum(lens, 1e-12)
    nrm /= lens[:, None]
    return nrm


# ----------------------------
# Export
# ----------------------------
def write_obj(path: str, v: np.ndarray, f: np.ndarray, vt: np.ndarray | None, vn: np.ndarray | None) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(
            "# watertight solid heightfield mesh (smooth base + undulation + pedestals)\n")
        fp.write(f"# V={v.shape[0]} F={f.shape[0]}\n\n")
        for p in v:
            fp.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

        if vt is not None:
            fp.write("\n")
            for t in vt:
                fp.write(f"vt {t[0]:.8f} {t[1]:.8f}\n")

        if vn is not None:
            fp.write("\n")
            for n in vn:
                fp.write(f"vn {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")

        fp.write("\n")
        use_vt = vt is not None
        use_vn = vn is not None
        for tri in f:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            if use_vt and use_vn:
                fp.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            elif use_vt and not use_vn:
                fp.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")
            elif (not use_vt) and use_vn:
                fp.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                fp.write(f"f {a} {b} {c}\n")


def write_ply_ascii(path: str, v: np.ndarray, f: np.ndarray, vn: np.ndarray | None) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        has_n = vn is not None
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            fp.write("property float nx\nproperty float ny\nproperty float nz\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\nend_header\n")

        if has_n:
            for p, n in zip(v, vn):
                fp.write(
                    f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n")
        else:
            for p in v:
                fp.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")

        for tri in f:
            fp.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=256,
                    help="grid resolution n x n (>=2)")
    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=1.0)
    ap.add_argument("--y_min", type=float, default=0.0)
    ap.add_argument("--y_max", type=float, default=1.0)

    # Base (square slab)
    ap.add_argument("--base_z", type=float, default=0.0, help="bottom plane z")
    ap.add_argument("--base_thickness", type=float, default=0.10,
                    help="slab thickness (base top = base_z + thickness)")

    # Scales
    ap.add_argument("--bump_scale", type=float, default=0.45,
                    help="scale for bumps+pedestals sum")
    ap.add_argument("--und_scale", type=float, default=0.06,
                    help="scale for global undulation (small)")

    # Edge behavior (keep boundary clean and flat on top)
    ap.add_argument("--edge_fade", type=float, default=0.08,
                    help="fade width near outer boundary to flatten top (0 disables)")

    # Export
    ap.add_argument("--out_obj", type=str, default="solid_smooth_v2.obj")
    ap.add_argument("--out_ply", type=str, default="solid_smooth_v2.ply")
    ap.add_argument("--no_obj", action="store_true")
    ap.add_argument("--no_ply", action="store_true")
    ap.add_argument("--with_uv", action="store_true", help="OBJ: write UVs")
    ap.add_argument("--with_normals", action="store_true",
                    help="write smooth vertex normals")

    args = ap.parse_args()
    n = int(args.n)
    if n < 2:
        raise ValueError("--n must be >= 2")

    xg, yg = build_grid(n, args.x_min, args.x_max, args.y_min, args.y_max)

    z_base_top = args.base_z + args.base_thickness

    # Components
    und = global_undulation(xg, yg)  # [-~1, ~1]
    bumps = bumps_with_pedestals(xg, yg)

    # Fade near boundary so top connects nicely to side walls on a flat rim
    mask = boundary_fade_mask(
        xg, yg, args.x_min, args.x_max, args.y_min, args.y_max, args.edge_fade)

    # Final top surface
    z_top = z_base_top + mask * \
        (args.und_scale * und + args.bump_scale * bumps)

    # Vertices: top then bottom
    V_top = np.stack([xg.reshape(-1), yg.reshape(-1),
                     z_top.reshape(-1)], axis=1)
    V_bot = np.stack([xg.reshape(-1), yg.reshape(-1),
                      np.full((n * n,), args.base_z, dtype=np.float64)], axis=1)
    v = np.vstack([V_top, V_bot])
    top_offset = 0
    bot_offset = n * n

    # Faces: top + bottom
    F_top = triangulate_grid(n, invert=False) + top_offset
    F_bot = triangulate_grid(n, invert=True) + bot_offset

    # Side walls
    loop = boundary_loop_indices(n)
    side = []
    L = loop.shape[0]
    for k in range(L):
        aT = int(loop[k]) + top_offset
        bT = int(loop[(k + 1) % L]) + top_offset
        aB = int(loop[k]) + bot_offset
        bB = int(loop[(k + 1) % L]) + bot_offset
        side.append((aT, bT, bB))
        side.append((aT, bB, aB))
    F_side = np.array(side, dtype=np.int64)

    f = np.vstack([F_top, F_bot, F_side])

    # UV
    vt = None
    if args.with_uv:
        ux = (v[:, 0] - args.x_min) / max(args.x_max - args.x_min, 1e-12)
        uy = (v[:, 1] - args.y_min) / max(args.y_max - args.y_min, 1e-12)
        vt = np.stack([ux, uy], axis=1)

    # Normals
    vn = None
    if args.with_normals:
        vn = compute_vertex_normals(v, f)

    # Write
    if not args.no_obj:
        write_obj(args.out_obj, v, f, vt=vt, vn=vn)
        print(f"[OK] OBJ: {args.out_obj}  V={v.shape[0]} F={f.shape[0]}")
    if not args.no_ply:
        write_ply_ascii(args.out_ply, v, f, vn=vn)
        print(f"[OK] PLY: {args.out_ply}  V={v.shape[0]} F={f.shape[0]}")

    print(f"z range: [{v[:,2].min():.6f}, {v[:,2].max():.6f}]")
    print("Watertight: top + bottom + side walls.")
    print("Smooth: C2 cutoff for each bump + pedestal, plus smooth global undulation.")


if __name__ == "__main__":
    main()
