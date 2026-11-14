"""Tiny software renderer for checking geometry: orthographic, point-splatted triangles, z-buffer, Lambert, optional
textures. Writes PNGs next to each other (front | side | back)."""
import numpy as np

import png

VIEWS = {  # right, up, towards the viewer (the model faces -Z, its left is +X)
    "front": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, -1.0])),
    "side": (np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([-1.0, 0, 0])),
    "back": (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
}
_BARY = None


def _bary(k=36):
    global _BARY
    if _BARY is None:
        g = []
        n = 8
        for i in range(n + 1):
            for j in range(n + 1 - i):
                g.append((i / n, j / n))
        g = np.array(g)
        _BARY = np.stack([g[:, 0], g[:, 1], 1 - g[:, 0] - g[:, 1]], axis=1)
    return _BARY


def render(items, view="front", size=(700, 900), bounds=None, joints=None, light=(0.4, 0.6, -0.7)):
    """items: list of dicts {positions (N,3), normals (N,3), tris (M,3), uv (N,2) or None, texture (H,W,4) or None,
    color (3,)}. joints: optional (K,3) points drawn as red dots."""
    right, up, toward = VIEWS[view]
    W, H = size
    allp = np.concatenate([it["positions"] for it in items])
    if bounds is None:
        lo, hi = allp.min(0), allp.max(0)
    else:
        lo, hi = bounds
    cx, cy = (lo + hi) / 2 @ right, (lo + hi) / 2 @ up
    span = max(np.abs((hi - lo) @ np.abs(right)), np.abs((hi - lo) @ up) * W / H) * 1.05
    scale = W / span
    zbuf = np.full(W * H, np.inf)
    img = np.full((W * H, 3), 40, np.float32)
    L = np.array(light, np.float64)
    L /= np.linalg.norm(L)
    B = _bary()
    for it in items:
        P, N, T = it["positions"].astype(np.float64), it["normals"].astype(np.float64), it["tris"]
        tp = P[T]  # (M,3,3)
        sp = np.einsum("bk,mkc->mbc", B, tp).reshape(-1, 3)
        sn = np.einsum("bk,mkc->mbc", B, N[T]).reshape(-1, 3)
        sn /= np.maximum(np.linalg.norm(sn, axis=1, keepdims=True), 1e-12)
        if it.get("texture") is not None and it.get("uv") is not None:
            tex = it["texture"]
            su = np.einsum("bk,mkc->mbc", B, it["uv"][T]).reshape(-1, 2)
            tx = np.clip((su[:, 0] % 1.0) * tex.shape[1], 0, tex.shape[1] - 1).astype(int)
            ty = np.clip((su[:, 1] % 1.0) * tex.shape[0], 0, tex.shape[0] - 1).astype(int)
            col = tex[ty, tx, :3].astype(np.float32)
        else:
            col = np.tile(np.array(it.get("color", (200, 200, 200)), np.float32), (len(sp), 1))
        shade = 0.35 + 0.65 * np.clip(sn @ L, 0, 1)
        facing = sn @ toward
        shade = np.where(facing < 0, shade * 0.5, shade)
        x = ((sp @ right) - cx) * scale + W / 2
        y = H / 2 - ((sp @ up) - cy) * scale
        z = -(sp @ toward)
        c_all = col * shade[:, None]
        for ox, oy in ((0, 0), (1, 0), (0, 1), (1, 1)):  # 2x2 splats close the gaps between samples
            xs, ys = x + ox, y + oy
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            idx = (ys[ok].astype(int) * W + xs[ok].astype(int))
            zz = z[ok]
            order = np.argsort(-zz)  # far first; the nearest is written last
            idx, zz = idx[order], zz[order]
            c = c_all[ok][order]
            closer = zz < zbuf[idx]
            img[idx[closer]] = c[closer]
            np.minimum.at(zbuf, idx, zz)
    img = img.reshape(H, W, 3)
    if joints is not None:
        for p in joints:
            x = int(((p @ right) - cx) * scale + W / 2)
            y = int(H / 2 - ((p @ up) - cy) * scale)
            img[max(0, y - 2):y + 3, max(0, x - 2):x + 3] = (255, 40, 40)
    return np.clip(img, 0, 255).astype(np.uint8)


def sheet(path, items, views=("front", "side", "back"), size=(600, 900), joints=None, bounds=None):
    imgs = [render(items, v, size, bounds=bounds, joints=joints) for v in views]
    png.write(path, np.concatenate(imgs, axis=1))
