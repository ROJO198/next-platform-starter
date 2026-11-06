#!/usr/bin/env python3
"""
Donauwelle (كعكة الدانوب) recipe reel — 4K vertical (2160x3840), 30 fps, 60 s.

Pure-Python motion-graphics renderer (Pillow + numpy) piped into ffmpeg.
Usage:
    python3 render.py --fonts ./fonts --out donauwelle_reel_4k.mp4 [--preview SECONDS]
"""
import argparse, math, os, subprocess, sys, wave
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2160, 3840
FPS = 30
DUR = 60.0
SS = 2                       # supersampling for illustrations
RY, RH = 1150, 1550          # illustration region (y offset, height)

# ---------- palette ----------
CREAM_BG1 = (255, 247, 236)
CREAM_BG2 = (246, 224, 198)
CHOC = (59, 32, 20)
CHOC_MID = (106, 62, 38)
CHERRY = (190, 22, 48)
GOLD = (232, 176, 75)
WHITE = (255, 255, 255)
VAN_RAW = (247, 228, 182)
VAN_BAKED = (234, 192, 110)
CHO_RAW = (128, 78, 48)
CHO_BAKED = (92, 52, 30)
BUTTERCREAM = (255, 248, 228)
GLAZE = (46, 26, 16)
TRAY = (138, 143, 152)

FONT_DIR = "fonts"

# ---------- easing ----------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out(t): t = clamp(t); return 1 - (1 - t) ** 3
def ease_in_out(t): t = clamp(t); return 3 * t * t - 2 * t * t * t
def ease_back(t):
    t = clamp(t); c1 = 1.70158; c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2
def bounce(t):
    t = clamp(t); n, d = 7.5625, 2.75
    if t < 1 / d: return n * t * t
    if t < 2 / d: t -= 1.5 / d; return n * t * t + .75
    if t < 2.5 / d: t -= 2.25 / d; return n * t * t + .9375
    t -= 2.625 / d; return n * t * t + .984375
def lerp(a, b, t): return a + (b - a) * t
def lerpc(c1, c2, t): return tuple(int(lerp(a, b, clamp(t))) for a, b in zip(c1, c2))
def prog(t, a, b): return clamp((t - a) / (b - a)) if b > a else float(t >= a)

# ---------- text (mixed Arabic / Latin, RTL) ----------
def is_arabic(tok): return any('؀' <= ch <= 'ۿ' for ch in tok)

@lru_cache(maxsize=64)
def font(kind, weight, size):
    if kind == "display":
        f = "lalezar-arabic-400-normal.woff" if True else ""
        lat = "lalezar-latin-400-normal.woff"
    else:
        f = f"cairo-arabic-{weight}-normal.woff"
        lat = f"cairo-latin-{weight}-normal.woff"
    return (ImageFont.truetype(os.path.join(FONT_DIR, f), size),
            ImageFont.truetype(os.path.join(FONT_DIR, lat), size))

@lru_cache(maxsize=512)
def text_img(text, size, color, weight=900, kind="body", shadow=True, stroke=0, stroke_color=None):
    fa, fl = font(kind, weight, size)
    toks = text.split(" ")
    pieces = []
    for tok in toks:
        if not tok:
            continue
        ar = is_arabic(tok)
        f = fa if ar else fl
        kw = dict(direction="rtl") if ar else {}
        bbox = f.getbbox(tok, stroke_width=stroke, **kw)
        pieces.append((tok, f, kw, bbox))
    gap = int(size * 0.28)
    widths = [b[2] - b[0] for *_, b in pieces]
    tw = sum(widths) + gap * (len(pieces) - 1)
    pad = int(size * 0.35)
    th = int(size * 1.9)
    img = Image.new("RGBA", (tw + pad * 2, th + pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x = pad + tw                         # RTL: start from the right edge
    base_y = pad // 2 + int(size * 0.2)
    for (tok, f, kw, b), w in zip(pieces, widths):
        x -= w
        d.text((x - b[0], base_y), tok, font=f, fill=color, stroke_width=stroke,
               stroke_fill=stroke_color, **kw)
        x -= gap
    bb = img.getbbox()
    if bb:
        img = img.crop((max(0, bb[0] - 8), max(0, bb[1] - 8), min(img.width, bb[2] + 8), min(img.height, bb[3] + 8)))
    if shadow:
        sh = Image.new("RGBA", (img.width + 60, img.height + 60), (0, 0, 0, 0))
        a = img.getchannel("A").point(lambda v: int(v * 0.28))
        blk = Image.new("RGBA", img.size, (40, 20, 10, 255)); blk.putalpha(a)
        sh.paste(blk, (30, 42), blk)
        sh = sh.filter(ImageFilter.GaussianBlur(14))
        sh.alpha_composite(img, (30, 30))
        img = sh
    return img

def paste(frame, img, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.01 or scale <= 0.01:
        return
    if abs(scale - 1) > 0.005:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    if alpha < 0.999:
        img = img.copy()
        img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha)))
    frame.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))

def text(frame, s, cx, cy, size, color=CHOC, alpha=1.0, scale=1.0, weight=900, kind="body", shadow=True,
         stroke=0, stroke_color=None):
    paste(frame, text_img(s, size, color, weight, kind, shadow, stroke, stroke_color), cx, cy, alpha, scale)

@lru_cache(maxsize=128)
def pill_img(label, size, bg, fg):
    t = text_img(label, size, fg, 900, "body", False)
    w, h = t.width + int(size * 1.2), int(size * 1.75)
    img = Image.new("RGBA", (w + 40, h + 40), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((20, 26, 20 + w, 26 + h), radius=h // 2, fill=(0, 0, 0, 50))
    img = img.filter(ImageFilter.GaussianBlur(8))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((20, 20, 20 + w, 20 + h), radius=h // 2, fill=bg)
    img.alpha_composite(t, (20 + (w - t.width) // 2, 20 + (h - t.height) // 2))
    return img

@lru_cache(maxsize=64)
def card_img(qty, name, accent):
    w, h = 900, 300
    img = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((30, 42, 30 + w, 42 + h), radius=60, fill=(80, 40, 20, 60))
    img = img.filter(ImageFilter.GaussianBlur(16))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((30, 30, 30 + w, 30 + h), radius=60, fill=(255, 255, 255, 250))
    d.rounded_rectangle((30 + w - 40, 70, 30 + w - 16, 30 + h - 40), radius=12, fill=accent)
    q = text_img(qty, 96, accent, 900, "body", False)
    n = text_img(name, 76, CHOC, 700, "body", False)
    img.alpha_composite(q, (30 + w - 80 - q.width, 30 + 22))
    img.alpha_composite(n, (30 + w - 80 - n.width, 30 + 150))
    return img

# ---------- background ----------
def make_background():
    y = np.linspace(0, 1, H)[:, None, None]
    c1 = np.array(CREAM_BG1, float); c2 = np.array(CREAM_BG2, float)
    bg = (c1 * (1 - y) + c2 * y) * np.ones((1, W, 1))
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(7)
    for _ in range(9):   # soft bokeh blobs
        cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(250, 650)
        col = np.array(rng.choice([GOLD, CHERRY, CHOC_MID]), float)
        m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None] * 0.07
        bg = bg * (1 - m) + col * m
    v = 1 - 0.18 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2)[..., None]   # vignette
    bg = bg * v + rng.normal(0, 1.4, bg.shape)                                         # fine grain
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).convert("RGBA")

# ---------- illustration helpers (region units, drawn at SS) ----------
X0, X1, YB = 260, 1900, 1300
HV, HC, HCR, HG = 230, 170, 220, 52
NCH = 12
CH_X = [X0 + 90 + i * (X1 - X0 - 180) / (NCH - 1) for i in range(NCH)]

def S(v): return v * SS
def P(pts): return [(S(x), S(y)) for x, y in pts]

def new_layer(): return Image.new("RGBA", (W * SS, RH * SS), (0, 0, 0, 0))

def cake_profile(st):
    """returns functions for the layer boundaries given state dict."""
    bake = st.get("bake", 0); rise = 1 + 0.12 * ease_in_out(bake)
    pv, pc = st.get("van", 0), st.get("choc", 0)
    def interface(x):
        mound = 0.25 * (1 - pv) * math.exp(-((x - 1080) / 380) ** 2)
        flat = YB - HV * pv * rise * (1 + mound)
        wave = 70 * ease_in_out(bake) * math.sin(2 * math.pi * (x - X0) / 410 + 0.6)
        return flat + wave * min(1, pv)
    def top(x):
        mound = 0.35 * (1 - pc) * math.exp(-((x - 1080) / 380) ** 2)
        return YB - (HV * pv + HC * pc * (1 + mound)) * rise - 10 * ease_in_out(bake) * math.exp(-((x - 1080) / 700) ** 2)
    return interface, top

def spread_poly(base_fn, h, spread, ripple=0.0, phase=0.0):
    hw = (X1 - X0) / 2 * ease_out(spread); cx = (X0 + X1) / 2
    if hw < 4: return None
    xs = np.linspace(cx - hw, cx + hw, 160)
    top, bot = [], []
    for x in xs:
        u = (x - cx) / hw
        taper = (1 - u * u) ** 0.35 if spread < 0.999 else 1.0
        b = base_fn(x)
        r = ripple * 7 * math.sin((x - X0) / 26 + phase)
        top.append((x, b - h * taper - r)); bot.append((x, b))
    return top + bot[::-1]

def draw_tray(d, cut_alpha=1):
    d.rectangle((S(X0 - 40), S(YB - 770), S(X0), S(YB + 40)), fill=TRAY)
    d.rectangle((S(X1), S(YB - 770), S(X1 + 40), S(YB + 40)), fill=TRAY)
    d.rectangle((S(X0 - 40), S(YB), S(X1 + 40), S(YB + 40)), fill=TRAY)
    d.rectangle((S(X0 - 40), S(YB - 770), S(X0 - 28), S(YB + 40)), fill=(178, 183, 190))
    d.rectangle((S(X1 + 28), S(YB - 770), S(X1 + 40), S(YB + 40)), fill=(110, 114, 122))

def draw_cherry(d, x, y, r=34, a=255):
    d.ellipse((S(x - r), S(y - r), S(x + r), S(y + r)), fill=CHERRY + (a,))
    d.ellipse((S(x - r * .45), S(y - r * .6), S(x - r * .05), S(y - r * .2)), fill=(255, 140, 150, int(a * .8)))

def draw_cake(layer, st):
    out_layer = layer
    layer = Image.new("RGBA", out_layer.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    interface, top = cake_profile(st)
    bake = st.get("bake", 0)
    xs = np.linspace(X0, X1, 200)
    if st.get("van", 0) > 0:
        d.polygon(P([(X0, YB)] + [(x, interface(x)) for x in xs] + [(X1, YB)]), fill=lerpc(VAN_RAW, VAN_BAKED, bake))
        # crumb texture
        rng = np.random.default_rng(3)
        for _ in range(int(140 * bake)):
            x = rng.uniform(X0 + 10, X1 - 10); y = rng.uniform(interface(x) + 12, YB - 8)
            d.ellipse((S(x - 4), S(y - 4), S(x + 4), S(y + 4)), fill=(222, 178, 105))
    if st.get("choc", 0) > 0:
        d.polygon(P([(x, interface(x)) for x in xs] + [(x, top(x)) for x in xs[::-1]]), fill=lerpc(CHO_RAW, CHO_BAKED, bake))
    # cherries
    cp = st.get("cherries", -1)
    if cp >= 0:
        for i, x in enumerate(CH_X):
            k = prog(cp, i / NCH * 0.6, i / NCH * 0.6 + 0.4)
            if k <= 0: continue
            rest = top(x) + 8
            sunk = interface(x) - 6 + 18 * math.sin(i * 1.7)
            y_end = lerp(rest, sunk, ease_in_out(bake))
            y = lerp(-200, y_end, bounce(k))
            draw_cherry(d, x, y, 34 - 3 * bake)
    ctop = lambda x: top(x)
    if st.get("cream", 0) > 0:
        poly = spread_poly(ctop, HCR * min(1, st["cream"] * 1.15), st["cream"])
        if poly: d.polygon(P(poly), fill=BUTTERCREAM)
    if st.get("glaze", 0) > 0:
        cbase = lambda x: top(x) - HCR
        poly = spread_poly(cbase, HG * min(1, st["glaze"] * 1.2), st["glaze"], ripple=st.get("comb", 0))
        if poly:
            d.polygon(P(poly), fill=GLAZE)
            pts = poly[:len(poly) // 2]
            d.line(P([(x, y + 7) for x, y in pts[4:-4]]), fill=(110, 70, 50), width=S(4))
    draw_tray(d)
    # knife + split
    cut = st.get("cut", 0)
    if cut > 0:
        kx = 1230
        sep = ease_in_out(prog(cut, 0.55, 1.0)) * 110
        if sep > 0:
            right = layer.crop((S(kx), 0, layer.width, layer.height))
            layer.paste((0, 0, 0, 0), (S(kx), 0, layer.width, layer.height))
            layer.alpha_composite(right, (int(S(kx + sep)), 0))
            d = ImageDraw.Draw(layer)
        ky = lerp(-260, YB - 20, ease_in_out(prog(cut, 0, 0.5)))
        ka = 1 - prog(cut, 0.55, 0.8)
        if ka > 0:
            a = int(255 * ka)
            d.polygon(P([(kx - 10, ky - 700), (kx + 14, ky - 700), (kx + 14, ky), (kx - 10, ky - 60)]), fill=(205, 212, 220, a))
            d.rectangle((S(kx - 22), S(ky - 980), S(kx + 26), S(ky - 700)), fill=(40, 30, 28, a))
    sh = Image.new("RGBA", layer.size, (0, 0, 0, 0)); sd = ImageDraw.Draw(sh)
    sd.ellipse((S(X0 - 120), S(YB + 10), S(X1 + 120 + 110 * (cut > 0)), S(YB + 140)), fill=(70, 40, 20, 90))
    out_layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(S(30))))
    out_layer.alpha_composite(layer)

def draw_pour(layer, x, y_from, y_to, color, width=46):
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((S(x - width / 2), S(y_from), S(x + width / 2), S(y_to)), radius=S(width / 2), fill=color)

def draw_bowl(d, cx, cy, rx, ry, content, swirl=0.0, fill_lvl=1.0):
    d.ellipse((S(cx - rx * 1.05), S(cy + ry * .95), S(cx + rx * 1.05), S(cy + ry * 1.25)), fill=(90, 50, 30, 50))
    d.pieslice((S(cx - rx), S(cy - ry), S(cx + rx), S(cy + ry)), 0, 180, fill=(232, 238, 243))
    if fill_lvl > 0:
        cyc = cy + ry * 0.12 * (1 - fill_lvl)
        d.ellipse((S(cx - rx * .9), S(cyc - ry * .16), S(cx + rx * .9), S(cyc + ry * .16)), fill=content)
        if swirl:
            for k in range(3):
                a0 = swirl * 360 + k * 120
                d.arc((S(cx - rx * .6), S(cyc - ry * .1), S(cx + rx * .6), S(cyc + ry * .1)), a0, a0 + 70,
                      fill=lerpc(content, WHITE, .45), width=S(6))
    d.ellipse((S(cx - rx), S(cy - ry * .18), S(cx + rx), S(cy + ry * .18)), outline=(205, 214, 222), width=S(14))

def draw_whisk(d, cx, cy, ang):
    x = cx + 120 * math.sin(ang)
    d.line((S(x + 160), S(cy - 620), S(x), S(cy - 120)), fill=(60, 60, 66), width=S(26))
    for k in range(-2, 3):
        d.ellipse((S(x - 70 + k * 12), S(cy - 160), S(x + 70 - k * 12), S(cy + 70)), outline=(185, 192, 200), width=S(7))

def finish(layer):
    return layer.reduce(SS)

# ---------- progress bar ----------
STAGES = [(11, 17), (17, 22), (22, 27), (27, 32), (32, 39), (39, 46), (46, 52)]
def progress_bar(fr, t):
    if t < 11 or t > 52.3: return
    d = ImageDraw.Draw(fr)
    n = len(STAGES); gw = 1700 / n; x0 = W / 2 + 850
    for i, (a, b) in enumerate(STAGES):
        xr = x0 - i * gw; xl = xr - gw + 18
        d.rounded_rectangle((xl, 230, xr, 252), radius=11, fill=(59, 32, 20, 40))
        p = prog(t, a, b)
        if p > 0:
            d.rounded_rectangle((xr - (xr - xl) * p, 230, xr, 252), radius=11, fill=CHERRY)

def header(fr, t, a, b, stage, title, sub=None, sub2=None):
    lt = t - a; out = 1 - prog(t, b - 0.3, b)
    pin = ease_back(prog(lt, 0, 0.45))
    paste(fr, pill_img(stage, 80, CHERRY, WHITE), W / 2, 420, alpha=out * clamp(pin), scale=0.6 + 0.4 * pin)
    tin = ease_out(prog(lt, 0.1, 0.6))
    text(fr, title, W / 2, 640 + 60 * (1 - tin), 150, CHOC, alpha=tin * out, kind="display")
    if sub:
        sin_ = ease_out(prog(lt, 0.3, 0.8))
        text(fr, sub, W / 2, 860 + 40 * (1 - sin_), 84, CHOC_MID, alpha=sin_ * out, weight=700)
    if sub2:
        s2 = ease_out(prog(lt, 0.45, 0.95))
        text(fr, sub2, W / 2, 990 + 40 * (1 - s2), 84, CHOC_MID, alpha=s2 * out, weight=700)

def caption(fr, t, s, a, b, color=CHOC):
    al = prog(t, a, a + 0.25) * (1 - prog(t, b - 0.25, b))
    if al <= 0: return
    img = text_img(s, 92, WHITE, 900, "body", False)
    if img.width > 1840:
        img = img.resize((1840, int(img.height * 1840 / img.width)), Image.LANCZOS)
    w, h = img.width + 120, img.height + 70
    box = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(box).rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=color + (235,))
    box.alpha_composite(img, ((w - img.width) // 2, (h - img.height) // 2))
    paste(fr, box, W / 2, 2950 + 30 * (1 - ease_out(prog(t, a, a + 0.3))), alpha=al, scale=0.92 + 0.08 * ease_back(prog(t, a, a + 0.35)))

def region(fr, layer_small, alpha=1.0, scale=1.0, dy=0):
    if alpha < 0.999:
        layer_small.putalpha(layer_small.getchannel("A").point(lambda v: int(v * alpha)))
    if abs(scale - 1) > 0.003:
        nw, nh = int(W * scale), int(RH * scale)
        layer_small = layer_small.resize((nw, nh), Image.BICUBIC)
        fr.alpha_composite(layer_small, (int((W - nw) / 2), int(RY + dy - (nh - RH) / 2)))
    else:
        fr.alpha_composite(layer_small, (0, int(RY + dy)))

def sparkle(fr, t, pts, color=GOLD):
    d = ImageDraw.Draw(fr)
    for i, (x, y) in enumerate(pts):
        k = (math.sin(t * 5 + i * 1.9) + 1) / 2
        r = 18 + 26 * k
        d.polygon([(x, y - r * 2), (x + r * .35, y - r * .35), (x + r * 2, y), (x + r * .35, y + r * .35),
                   (x, y + r * 2), (x - r * .35, y + r * .35), (x - r * 2, y), (x - r * .35, y - r * .35)],
                  fill=color + (int(140 + 115 * k),))

FULL = dict(van=1, choc=1, cherries=1, bake=1, cream=1, glaze=1, comb=1)

# ---------- scenes ----------
def s_hook(fr, t):                       # 0–3
    lt = t
    cut = prog(lt, 0.2, 1.6)
    L = new_layer(); st = dict(FULL); st["cut"] = 0.55 + 0.45 * cut
    draw_cake(L, st)
    sc = 1.15 - 0.1 * ease_out(prog(lt, 0, 3))
    region(fr, finish(L), alpha=ease_out(prog(lt, 0, .4)) * (1 - prog(lt, 2.75, 3)), scale=sc, dy=160)
    o = 1 - prog(lt, 2.75, 3)
    k = ease_back(prog(lt, 0.05, 0.55))
    text(fr, "كعكة الدانوب", W / 2, 640, 230, CHOC, alpha=o * clamp(k), scale=0.7 + 0.3 * k, kind="display")
    paste(fr, pill_img("Donauwelle  •  ألمانيا", 74, CHOC, GOLD), W / 2, 900, alpha=o * prog(lt, .4, .8))
    tin = ease_out(prog(lt, 0.8, 1.3))
    text(fr, "سرّ الأمواج المخفية بداخلها", W / 2, 3020 + 50 * (1 - tin), 104, CHERRY, alpha=tin * o)
    text(fr, "في 60 ثانية", W / 2, 3180 + 50 * (1 - tin), 92, CHOC_MID, alpha=tin * o, weight=700)
    if lt > 1.5:
        sparkle(fr, t, [(560, 1700), (1660, 1580), (1880, 2350)])

ING1 = [("250 غ", "زبدة طرية", GOLD), ("250 غ", "سكر", CHOC_MID), ("5", "بيضات", GOLD),
        ("375 غ", "دقيق", CHOC_MID), ("15 غ", "بيكنج باودر", GOLD), ("3 م.ك", "كاكاو + 2 م.ك حليب", CHOC_MID),
        ("700 غ", "كرز حامض مصفّى", CHERRY), ("1", "رشة ملح + فانيلا", GOLD)]
ING2 = [("1 لتر", "حليب", GOLD), ("2 كيس", "بودينغ فانيلا", GOLD), ("100 غ", "سكر", CHOC_MID),
        ("250 غ", "زبدة طرية", GOLD), ("200 غ", "شوكولاتة داكنة", CHOC), ("2 م.ك", "زيت جوز الهند", CHOC_MID)]

def s_ingredients(fr, t, a, b, title, sub, items):
    lt = t - a; out = 1 - prog(t, b - 0.3, b)
    tin = ease_out(prog(lt, 0, .45))
    paste(fr, pill_img("المكوّنات", 80, CHOC, GOLD), W / 2, 420, alpha=tin * out)
    text(fr, title, W / 2, 640 + 50 * (1 - tin), 150, CHOC, alpha=tin * out, kind="display")
    text(fr, sub, W / 2, 840, 76, CHOC_MID, alpha=tin * out, weight=700)
    for i, (q, n, c) in enumerate(items):
        col = i % 2; row = i // 2
        cx = W / 2 + (500 if col == 0 else -500); cy = 1180 + row * 380
        k = ease_back(prog(lt, 0.35 + i * 0.22, 0.75 + i * 0.22))
        paste(fr, card_img(q, n, c), cx, cy + 60 * (1 - clamp(k)), alpha=clamp(k) * out, scale=0.75 + 0.25 * k)

def s_stage1(fr, t):                     # 11–17 mixing
    a, b = 11, 17; lt = t - a
    header(fr, t, a, b, "المرحلة 1", "العجينة الأساسية", "اخفقي الزبدة مع السكر حتى تصبح كريمية")
    L = new_layer(); d = ImageDraw.Draw(L)
    drops = [("زبدة", 0.6), ("سكر", 1.7), ("بيض", 2.8), ("دقيق", 3.9)]
    lvl = 0.35 + 0.65 * prog(lt, 0.6, 5)
    draw_bowl(d, 1080, 860, 640, 430, VAN_RAW, swirl=lt * 0.9, fill_lvl=lvl)
    for i, (nm, t0) in enumerate(drops):
        k = prog(lt, t0, t0 + 0.6)
        if 0 < k < 1:
            y = lerp(-150, 820, ease_in_out(k)); x = 1080 + (i - 1.5) * 90
            if i == 0: d.rounded_rectangle((S(x - 90), S(y - 70), S(x + 90), S(y + 70)), radius=S(20), fill=(255, 222, 120))
            elif i == 1:
                for j in range(14): d.ellipse((S(x - 80 + (j * 37) % 160), S(y - 60 + (j * 53) % 120), S(x - 66 + (j * 37) % 160), S(y - 46 + (j * 53) % 120)), fill=WHITE)
            elif i == 2: d.ellipse((S(x - 70), S(y - 90), S(x + 70), S(y + 90)), fill=(250, 238, 220), outline=(220, 200, 170), width=S(5))
            else: d.pieslice((S(x - 140), S(y - 80), S(x + 140), S(y + 140)), 180, 360, fill=(252, 250, 244))
    draw_whisk(d, 1080, 860, lt * 9)
    region(fr, finish(L), alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)))
    caption(fr, t, "250 غ زبدة + 250 غ سكر + فانيلا", 11.4, 13.3)
    caption(fr, t, "أضيفي 5 بيضات واحدة تلو الأخرى", 13.3, 14.9)
    caption(fr, t, "ثم 375 غ دقيق + 15 غ بيكنج باودر", 14.9, 17)

def s_stage2(fr, t):                     # 17–22 layering
    a, b = 17, 22; lt = t - a
    header(fr, t, a, b, "المرحلة 2", "طبقتان من العجين", "نصف العجينة فاتح، والنصف الثاني بالكاكاو")
    van = ease_in_out(prog(lt, 0.4, 2.4)); choc = ease_in_out(prog(lt, 2.7, 4.6))
    L = new_layer()
    interface, top = cake_profile(dict(van=van, choc=choc))
    if 0.4 < lt < 2.5: draw_pour(L, 1080, -50, interface(1080) + 10, VAN_RAW)
    if 2.7 < lt < 4.7: draw_pour(L, 1080, -50, top(1080) + 10, CHO_RAW)
    draw_cake(L, dict(van=van, choc=choc))
    region(fr, finish(L), alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)))
    caption(fr, t, "افردي العجينة الفاتحة في صينية 40×30 سم", 17.3, 19.6)
    caption(fr, t, "3 م.ك كاكاو + 2 م.ك حليب للباقي", 19.6, 22, CHOC_MID)

def s_stage3(fr, t):                     # 22–27 cherries
    a, b = 22, 27; lt = t - a
    header(fr, t, a, b, "المرحلة 3", "الكرز الحامض", "وزّعيه واضغطيه بخفة داخل العجين")
    L = new_layer(); draw_cake(L, dict(van=1, choc=1, cherries=prog(lt, 0.4, 3.6)))
    region(fr, finish(L), alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)))
    caption(fr, t, "700 غ كرز مصفّى جيدًا من العصير", 22.4, 24.8, CHERRY)
    caption(fr, t, "نصيحة - جففيه بمنديل حتى لا يغرق", 24.8, 27, CHOC_MID)

def s_stage4(fr, t):                     # 27–32 bake
    a, b = 27, 32; lt = t - a
    header(fr, t, a, b, "المرحلة 4", "الخَبز = سرّ الأمواج", "الكرز يغوص فيرسم الموجة بين الطبقتين")
    bake = ease_in_out(prog(lt, 0.6, 4.2))
    L = new_layer()
    glow = Image.new("RGBA", L.size, (0, 0, 0, 0)); gd = ImageDraw.Draw(glow)
    gd.ellipse((S(150), S(250), S(2010), S(1500)), fill=(255, 140, 40, int(110 * math.sin(math.pi * prog(lt, 0.2, 4.8)))))
    L.alpha_composite(glow.filter(ImageFilter.GaussianBlur(S(80))))
    d = ImageDraw.Draw(L)
    for k in range(4):   # heat lines
        x = 600 + k * 320; ph = lt * 4 + k
        pts = [(x + 30 * math.sin(ph + j * 0.5), 560 - j * 40) for j in range(10)]
        d.line(P(pts), fill=(232, 120, 40, int(160 * math.sin(math.pi * prog(lt, 0.3, 4.6)))), width=S(10))
    draw_cake(L, dict(van=1, choc=1, cherries=1, bake=bake))
    # timer ring
    d = ImageDraw.Draw(L); cx, cy, r = 1080, 230, 150
    d.ellipse((S(cx - r), S(cy - r), S(cx + r), S(cy + r)), fill=(255, 255, 255, 235), outline=(240, 220, 200), width=S(16))
    d.arc((S(cx - r), S(cy - r), S(cx + r), S(cy + r)), -90, -90 + 360 * bake, fill=CHERRY, width=S(16))
    region(fr, finish(L), alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)))
    mins = int(round(40 * bake))
    text(fr, f"{mins}", W / 2, RY + 225, 120, CHOC, shadow=False, alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)))
    caption(fr, t, "180°C فرن مسخّن • 40 دقيقة تقريبًا", 27.4, 30, CHOC)
    caption(fr, t, "اتركيها تبرد تمامًا قبل الكريمة", 30, 32, CHOC_MID)

def s_stage5(fr, t):                     # 32–39 buttercream
    a, b = 32, 39; lt = t - a
    header(fr, t, a, b, "المرحلة 5", "كريمة الزبدة بالبودينغ", "اطبخي البودينغ مع الحليب والسكر وبرّديه")
    L = new_layer(); d = ImageDraw.Draw(L)
    if lt < 3.6:
        al = prog(lt, 0, .3) * (1 - prog(lt, 3.3, 3.6))
        merge = ease_in_out(prog(lt, 2.0, 3.3))
        draw_bowl(d, lerp(560, 1080, merge), 820, 380, 270, (255, 214, 90))
        draw_bowl(d, lerp(1600, 1080, merge), 820, 380, 270, (255, 236, 170))
        region(fr, finish(L), alpha=al)
        if merge < 0.5:
            ta = al * (1 - prog(merge, 0.2, 0.5))
            pill1 = pill_img("بودينغ 20°C", 70, (255, 214, 90), CHOC); pill2 = pill_img("زبدة 20°C", 70, (255, 236, 170), CHOC)
            paste(fr, pill1, 560, RY + 1260, alpha=ta); paste(fr, pill2, 1600, RY + 1260, alpha=ta)
            text(fr, "=", W / 2, RY + 900, 200, CHERRY, alpha=ta)
    else:
        cream = ease_in_out(prog(lt, 3.9, 6.3))
        st = dict(van=1, choc=1, cherries=1, bake=1, cream=cream)
        interface, top = cake_profile(st)
        if 3.9 < lt < 6.2: draw_pour(L, 1080, -50, top(1080) - HCR * min(1, cream * 1.15) + 10, BUTTERCREAM, 70)
        draw_cake(L, st)
        region(fr, finish(L), alpha=prog(lt, 3.6, 3.9) * (1 - prog(t, b - .3, b)))
    caption(fr, t, "1 لتر حليب + 2 كيس بودينغ + 100 غ سكر", 32.4, 34.5, GOLD)
    caption(fr, t, "السرّ: نفس الحرارة وإلا تتكتّل الكريمة", 34.5, 36.5, CHERRY)
    caption(fr, t, "اخفقي 250 غ زبدة ثم أضيفي البودينغ ملعقة ملعقة", 36.5, 39, CHOC_MID)

def s_stage6(fr, t):                     # 39–46 glaze + comb (top view)
    a, b = 39, 46; lt = t - a
    header(fr, t, a, b, "المرحلة 6", "الصوص والأمواج", "200 غ شوكولاتة + 2 م.ك زيت جوز الهند")
    L = new_layer(); d = ImageDraw.Draw(L)
    x0, x1, y0, y1 = 300, 1860, 230, 1300
    d.rounded_rectangle((S(x0 - 40), S(y0 - 40), S(x1 + 40), S(y1 + 40)), radius=S(30), fill=TRAY)
    d.rectangle((S(x0), S(y0), S(x1), S(y1)), fill=BUTTERCREAM)
    pour = ease_out(prog(lt, 0.4, 2.6))
    if pour > 0:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * pour, (y1 - y0) / 2 * pour
        d.rounded_rectangle((S(cx - hw), S(cy - hh), S(cx + hw), S(cy + hh)), radius=S(max(2, 300 * (1 - pour))), fill=GLAZE)
        d.ellipse((S(cx - hw * .5), S(cy - hh * .5), S(cx - hw * .1), S(cy - hh * .35)), fill=(90, 58, 40))
    comb = prog(lt, 3.0, 6.2)
    if comb > 0:
        xh = lerp(x0, x1, ease_in_out(comb))
        for j in range(8):
            yj = y0 + 80 + j * (y1 - y0 - 160) / 7
            pts = [(x, yj + 34 * math.sin(2 * math.pi * (x - x0) / 300)) for x in np.arange(x0, xh, 8)]
            if len(pts) > 1:
                d.line(P([(x, y + 6) for x, y in pts]), fill=(20, 10, 6), width=S(12))
                d.line(P(pts), fill=(120, 80, 58), width=S(10))
        ycomb = 34 * math.sin(2 * math.pi * (xh - x0) / 300)
        d.rounded_rectangle((S(xh - 22), S(y0 - 30 + ycomb), S(xh + 22), S(y1 + 30 + ycomb)), radius=S(14), fill=(230, 232, 236))
        d.rounded_rectangle((S(xh - 22), S(y0 - 160 + ycomb), S(xh + 22), S(y0 - 30 + ycomb)), radius=S(14), fill=(40, 30, 28))
    region(fr, finish(L), alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)))
    caption(fr, t, "اسكبي الشوكولاتة الذائبة وافرديها بسرعة", 39.4, 42, CHOC)
    caption(fr, t, "ارسمي الأمواج بمشط الكيك أو شوكة", 42, 46, CHERRY)

def s_stage7(fr, t):                     # 46–52 chill + cut
    a, b = 46, 52; lt = t - a
    header(fr, t, a, b, "المرحلة 7", "التبريد والتقطيع", "ساعتان في الثلاجة على الأقل")
    L = new_layer(); draw_cake(L, dict(FULL, cut=prog(lt, 1.2, 4.6)))
    z = 1 + 0.08 * ease_in_out(prog(lt, 3.6, 6))
    region(fr, finish(L), alpha=prog(lt, 0, .3) * (1 - prog(t, b - .3, b)), scale=z)
    caption(fr, t, "قطّعي بسكين ساخنة لحواف نظيفة", 46.4, 49.2, CHOC)
    caption(fr, t, "وهذه هي موجات الدانوب", 49.2, 52, CHERRY)
    if lt > 4.2: sparkle(fr, t, [(1120, 1750), (1500, 1900), (700, 2250)])

LEVELS = [("المستوى 1", "الكلاسيكية الألمانية • كرز وفانيلا", CHOC),
          ("المستوى 2", "ماسكاربوني بدل الزبدة • أخف وأطرى", GOLD),
          ("المستوى 3", "فستق حلبي + توت أحمر • نسخة فاخرة", CHERRY),
          ("المستوى 4", "كاسات دانوب للحفلات • بدون تقطيع", CHOC_MID)]

@lru_cache(maxsize=8)
def level_card(i):
    lvl, desc, col = LEVELS[i]
    w, h = 1840, 320
    img = Image.new("RGBA", (w + 60, h + 60), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    d.rounded_rectangle((30, 44, 30 + w, 44 + h), radius=70, fill=(80, 40, 20, 60))
    img = img.filter(ImageFilter.GaussianBlur(16)); d = ImageDraw.Draw(img)
    d.rounded_rectangle((30, 30, 30 + w, 30 + h), radius=70, fill=WHITE)
    pill = pill_img(lvl, 66, col, WHITE)
    img.alpha_composite(pill, (30 + w - pill.width - 30, 30 + (h - pill.height) // 2))
    tx = text_img(desc, 74, CHOC, 700, "body", False)
    maxw = w - pill.width - 80 - 300
    if tx.width > maxw:
        tx = tx.resize((maxw, int(tx.height * maxw / tx.width)), Image.LANCZOS)
    img.alpha_composite(tx, (30 + w - pill.width - 50 - tx.width, 30 + (h - tx.height) // 2))
    for k in range(i + 1):   # stars
        sx, sy, r = 110 + k * 70, 30 + h // 2, 26
        pts = [(sx + r * (1 if j % 2 == 0 else .45) * math.cos(math.pi / 2 + j * math.pi / 5),
                sy - r * (1 if j % 2 == 0 else .45) * math.sin(math.pi / 2 + j * math.pi / 5)) for j in range(10)]
        d = ImageDraw.Draw(img); d.polygon(pts, fill=GOLD)
    return img

def s_levels(fr, t):                     # 52–57
    a, b = 52, 57; lt = t - a; out = 1 - prog(t, b - .3, b)
    tin = ease_out(prog(lt, 0, .45))
    paste(fr, pill_img("طوّري الوصفة", 80, CHERRY, WHITE), W / 2, 420, alpha=tin * out)
    text(fr, "4 مستويات للدانوب", W / 2, 640 + 50 * (1 - tin), 150, CHOC, alpha=tin * out, kind="display")
    text(fr, "ابدئي بالكلاسيكية، ثم ارفعي المستوى", W / 2, 840, 80, CHOC_MID, alpha=tin * out, weight=700)
    for i in range(4):
        k = ease_back(prog(lt, 0.4 + i * 0.45, 0.85 + i * 0.45))
        paste(fr, level_card(i), W / 2 + 300 * (1 - clamp(k)), 1250 + i * 420, alpha=clamp(k) * out)
    caption(fr, t, "اكتبي في التعليقات - أي مستوى ستجرّبين؟", 53.6, 57, CHERRY)

def s_cta(fr, t):                        # 57–60
    lt = t - 57
    L = new_layer(); draw_cake(L, dict(FULL, cut=1))
    region(fr, finish(L), alpha=ease_out(prog(lt, 0, .4)), scale=0.9, dy=250)
    k = ease_back(prog(lt, 0.0, 0.5))
    text(fr, "احفظي الوصفة", W / 2, 620, 190, CHOC, alpha=clamp(k), scale=.7 + .3 * k, kind="display")
    # bookmark icon
    d = ImageDraw.Draw(fr); bx, by = W / 2, 930; s = clamp(ease_back(prog(lt, .3, .8)))
    if s > 0:
        d.polygon([(bx - 70 * s, by - 90 * s), (bx + 70 * s, by - 90 * s), (bx + 70 * s, by + 90 * s), (bx, by + 40 * s), (bx - 70 * s, by + 90 * s)], fill=CHERRY)
    paste(fr, pill_img("تابعيني لوصفات ريلز أسبوعية", 84, CHERRY, WHITE), W / 2, 2980, alpha=prog(lt, .5, .9), scale=0.9 + 0.1 * ease_back(prog(lt, .5, 1.0)))
    text(fr, "Donauwelle  •  كعكة الدانوب", W / 2, 3180, 76, CHOC_MID, alpha=prog(lt, .7, 1.1), weight=700)
    sparkle(fr, t, [(420, 1500), (1760, 1650), (1900, 2400), (300, 2500)])

SCENES = [
    (0, 3, s_hook),
    (3, 7.3, lambda fr, t: s_ingredients(fr, t, 3, 7.3, "الكيك", "صينية 40×30 سم • 20 قطعة", ING1)),
    (7.3, 11, lambda fr, t: s_ingredients(fr, t, 7.3, 11, "الكريمة والصوص", "لطبقة الكريمة والشوكولاتة", ING2)),
    (11, 17, s_stage1), (17, 22, s_stage2), (22, 27, s_stage3), (27, 32, s_stage4),
    (32, 39, s_stage5), (39, 46, s_stage6), (46, 52, s_stage7), (52, 57, s_levels), (57, 60.01, s_cta),
]

BG = None
def render_frame(i):
    global BG
    if BG is None: BG = make_background()
    t = i / FPS
    fr = BG.copy()
    for a, b, fn in SCENES:
        if a <= t < b:
            fn(fr, t)
    progress_bar(fr, t)
    # small brand mark (top-left, outside Reels UI)
    text(fr, "Donauwelle", 330, 120, 54, (59, 32, 20), alpha=0.55, weight=700, shadow=False)
    return fr.convert("RGB")

# ---------- encoding ----------
def render_chunk(args):
    idx, f0, f1, out = args
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "5.2", "-x264-params", "keyint=60:min-keyint=60:scenecut=0", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(f0, f1):
        p.stdin.write(render_frame(i).tobytes())
        if (i - f0) % 60 == 0:
            print(f"[chunk {idx}] frame {i}/{f1}", flush=True)
    p.stdin.close(); p.wait()
    return out

# ---------- audio ----------
SR = 48000
def synth_music(path, dur=DUR):
    n = int(SR * dur); t = np.arange(n) / SR
    mix = np.zeros(n)
    bpm = 100; beat = 60 / bpm
    def note(m): return 440 * 2 ** ((m - 69) / 12)
    prog_ch = [[60, 64, 67, 72], [55, 59, 62, 67], [57, 60, 64, 69], [53, 57, 60, 65]]   # C G Am F
    bar = beat * 4
    # pad
    for k in range(int(dur / bar) + 1):
        ch = prog_ch[k % 4]; s0 = int(k * bar * SR); s1 = min(n, int((k + 1) * bar * SR))
        if s0 >= n: break
        tt = np.arange(s1 - s0) / SR
        env = np.minimum(1, tt / 0.3) * np.minimum(1, (bar - tt) / 0.3)
        for m in ch:
            f = note(m - 12)
            mix[s0:s1] += 0.035 * env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) + 0.12 * np.sin(6 * np.pi * f * tt))
        # bass
        fb = note(ch[0] - 24)
        mix[s0:s1] += 0.09 * env * np.sin(2 * np.pi * fb * tt) * (0.6 + 0.4 * np.sin(2 * np.pi * tt / beat) ** 2)
    # pluck arpeggio (eighths) from 3 s
    step = beat / 2; k = 0
    while k * step < dur - 1.5:
        st = k * step
        if st >= 2.8:
            ch = prog_ch[int(st / bar) % 4]; m = ch[[0, 2, 1, 3, 2, 1, 3, 2][k % 8]] + 12
            s0 = int(st * SR); L = int(0.45 * SR); s1 = min(n, s0 + L); tt = np.arange(s1 - s0) / SR
            f = note(m)
            mix[s0:s1] += 0.05 * np.exp(-tt * 9) * (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt))
        k += 1
    rng = np.random.default_rng(1)
    # drums from 3 s
    b = 0
    while b * beat < dur - 1.5:
        st = b * beat
        if st >= 3:
            s0 = int(st * SR)
            L = int(0.3 * SR); tt = np.arange(L) / SR; s1 = min(n, s0 + L)
            kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-tt * 25)) * tt) * np.exp(-tt * 9) * 0.32
            mix[s0:s1] += kick[:s1 - s0]
            if b % 2 == 1:
                L2 = int(0.18 * SR); nz = rng.normal(0, 1, L2) * np.exp(-np.arange(L2) / SR * 22) * 0.09
                mix[s0:s0 + L2] += nz[:max(0, min(L2, n - s0))]
            for half in (0, 0.5):
                h0 = int((st + half * beat) * SR); L3 = int(0.05 * SR)
                if h0 + L3 < n:
                    hn = rng.normal(0, 1, L3); hn = np.diff(np.concatenate([[0], hn]))
                    mix[h0:h0 + L3] += hn * np.exp(-np.arange(L3) / SR * 80) * 0.03
        b += 1
    # whooshes at scene changes
    for st in [3, 7.3, 11, 17, 22, 27, 32, 39, 46, 52, 57]:
        L = int(0.5 * SR); s0 = int((st - 0.3) * SR); tt = np.arange(L) / SR
        nz = rng.normal(0, 1, L); nz = np.convolve(nz, np.ones(12) / 12, "same")
        mix[s0:s0 + L] += nz * np.sin(np.pi * tt / 0.5) ** 2 * 0.22
    # pops for ingredient cards and cherries
    pops = [3 + 0.35 + i * 0.22 for i in range(8)] + [7.3 + 0.35 + i * 0.22 for i in range(6)] + \
           [22 + 0.4 + (i / NCH * 0.6 + 0.4 * 0.6) * 3.2 for i in range(NCH)] + [52 + 0.4 + i * 0.45 for i in range(4)]
    for st in pops:
        s0 = int(st * SR); L = int(0.08 * SR); tt = np.arange(L) / SR
        mix[s0:s0 + L] += np.sin(2 * np.pi * (900 - 4000 * tt) * tt) * np.exp(-tt * 50) * 0.12
    # hook hit + ending
    tt = np.arange(int(1.5 * SR)) / SR
    hit = np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 3) * 0.3
    mix[:len(hit)] += hit
    fade = np.ones(n); fl = int(1.2 * SR); fade[-fl:] = np.linspace(1, 0, fl); fade[:int(.05 * SR)] = np.linspace(0, 1, int(.05 * SR))
    mix *= fade
    mix = np.tanh(mix * 1.4) / np.tanh(1.4)
    mix = mix / (np.max(np.abs(mix)) + 1e-9) * 0.89
    st = np.stack([mix, np.roll(mix, int(0.012 * SR)) * 0.97], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())

def main():
    global FONT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts", default="fonts")
    ap.add_argument("--out", default="donauwelle_reel_4k.mp4")
    ap.add_argument("--work", default="build")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--stills", nargs="*", type=float, help="only export PNG stills at these seconds")
    a = ap.parse_args()
    FONT_DIR = a.fonts
    os.makedirs(a.work, exist_ok=True)
    if a.stills:
        for s in a.stills:
            render_frame(int(s * FPS)).resize((W // 4, H // 4), Image.LANCZOS).save(os.path.join(a.work, f"still_{s:05.1f}.png"))
        return
    total = int(DUR * FPS)
    n = a.jobs; per = math.ceil(total / n)
    chunks = [(k, k * per, min(total, (k + 1) * per), os.path.join(a.work, f"chunk_{k}.mp4")) for k in range(n)]
    with Pool(n, initializer=_init, initargs=(a.fonts,)) as pool:
        outs = pool.map(render_chunk, chunks)
    lst = os.path.join(a.work, "list.txt")
    with open(lst, "w") as f:
        for o in outs: f.write(f"file '{os.path.abspath(o)}'\n")
    wav = os.path.join(a.work, "music.wav"); synth_music(wav)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart", a.out], check=True)
    print("done:", a.out)

def _init(fd):
    global FONT_DIR
    FONT_DIR = fd

if __name__ == "__main__":
    main()
