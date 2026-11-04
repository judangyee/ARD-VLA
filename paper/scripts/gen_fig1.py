"""Fig.1 teaser generator (rewrite). SVG units == pt at final 8.5cm print width.
All text widths are measured via real font metrics (Liberation Sans, metric-compatible
with Arial) so box/panel sizes are computed from actual glyph extents, not guesses.
"""
import math
import os
import sys
from PIL import ImageFont

BOLD_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
REG_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
_font_cache = {}

def _font(fs_pt, bold):
    key = (round(fs_pt * 10), bold)
    if key not in _font_cache:
        # render at 10x for sub-pixel precision, then scale back down
        _font_cache[key] = ImageFont.truetype(BOLD_PATH if bold else REG_PATH, max(1, round(fs_pt * 10)))
    return _font_cache[key]

def text_w(s, fs_pt, bold=True):
    f = _font(fs_pt, bold)
    bbox = f.getbbox(s)
    return (bbox[2] - bbox[0]) / 10.0

# ascent/descent ratios are ~constant across sizes for a given TTF
_ASCENT_RATIO, _DESCENT_RATIO = 0.905, 0.212

def text_top(baseline_y, fs_pt):
    return baseline_y - fs_pt * _ASCENT_RATIO

def text_bottom(baseline_y, fs_pt):
    return baseline_y + fs_pt * _DESCENT_RATIO

PRINT_W_CM = 8.5
PRINT_W_PT = PRINT_W_CM * 28.3465
TARGET_PT = 8.15  # small safety buffer over the 8pt floor

BLUE, ORANGE, GRAY, DARK, RED, GREEN = "#0072B2", "#E69F00", "#999999", "#1A1A1A", "#D55E00", "#009E73"

# ---- fixed (non-FS-scaling) layout constants, in pt ----
OUTER_MARGIN = 4
PANEL_GAP = 13
LABEL_PAD = 4
TITLE_PAD = 4
BOX_PAD = 3.5
ROLE_GAP = 6
ROLE_INNER_MARGIN = 3

TITLE1 = ["Human", "bimanual use"]
TITLE2 = ["Symmetric VLA", "(baseline)"]
TITLE3 = ["ARD-VLA", "(ours)"]
LBL_STAB_HOLD = "Stabilizer: hold"
LBL_ACT_TWIST = "Actuator: twist"


def panel1_w(fs):
    return max(text_w(LBL_STAB_HOLD, fs, True), text_w(LBL_ACT_TWIST, fs, True)) + 2 * LABEL_PAD


def panel2_w(fs):
    title_w = max(text_w(t, fs, True) for t in TITLE2)
    box_w = text_w("VLA policy", fs, False) + 2 * BOX_PAD + 2 * TITLE_PAD
    arm_w = max(text_w("Left arm", fs, False), text_w("Right arm", fs, False)) + 2 * TITLE_PAD
    return max(title_w + 2 * TITLE_PAD, box_w, 2 * arm_w + 2 * TITLE_PAD)


def panel3_w(fs):
    title_w = max(text_w(t, fs, True) for t in TITLE3)
    role_w = (2 * ROLE_INNER_MARGIN + text_w("Stabilizer", fs, False) + 2 * BOX_PAD
              + ROLE_GAP + text_w("Actuator", fs, False) + 2 * BOX_PAD)
    return max(title_w + 2 * TITLE_PAD, role_w)


def total_w(fs):
    return panel1_w(fs) + panel2_w(fs) + panel3_w(fs) + 2 * PANEL_GAP + 2 * OUTER_MARGIN


def solve_fs():
    lo, hi = 1.0, 60.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ratio = mid * PRINT_W_PT / total_w(mid)
        if ratio < TARGET_PT:
            lo = mid
        else:
            hi = mid
    return hi


FS = solve_fs()
FS_TITLE = FS  # unified size; titles are short enough not to be the bottleneck
assert FS * PRINT_W_PT / total_w(FS) >= 8.0, "font-size solve failed to clear 8pt floor"

P1W, P2W, P3W = panel1_w(FS), panel2_w(FS), panel3_w(FS)
W = total_w(FS)
panels_x = [OUTER_MARGIN, OUTER_MARGIN + P1W + PANEL_GAP, OUTER_MARGIN + P1W + PANEL_GAP + P2W + PANEL_GAP]
panel_w = [P1W, P2W, P3W]

PANEL_Y = FS * 3.6  # room for 2-line title above
PANEL_H = FS * 16.0
H = PANEL_Y + PANEL_H + FS * 1.2

print(f"FS={FS:.3f} finalpt={FS*PRINT_W_PT/W:.3f} W={W:.2f} H={H:.2f}")
print(f"panel widths: {P1W:.1f} {P2W:.1f} {P3W:.1f}")


def rect(x, y, w, h, fill, stroke, sw=1.0, rx=None):
    rxattr = f' rx="{rx}"' if rx is not None else ""
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"{rxattr} fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def text(x, y, s, fs, fill, bold=True, anchor="middle"):
    weight = "bold" if bold else "normal"
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{fs:.3f}" '
            f'font-family="Arial,Helvetica,sans-serif" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{s}</text>')


_MARKER_IDS = {}

def _marker_id(color):
    """cairosvg doesn't support fill=context-stroke on markers, so make one marker per color."""
    if color not in _MARKER_IDS:
        _MARKER_IDS[color] = f"arrow_{len(_MARKER_IDS)}"
    return _MARKER_IDS[color]


def marker_defs():
    out = []
    ms = FS * 0.34
    for color, mid in _MARKER_IDS.items():
        out.append(f'<marker id="{mid}" markerWidth="{ms:.2f}" markerHeight="{ms:.2f}" refX="{ms*0.75:.2f}" refY="{ms/2:.2f}" orient="auto">'
                    f'<path d="M0,0 L{ms:.2f},{ms/2:.2f} L0,{ms:.2f} Z" fill="{color}"/></marker>')
    return "\n".join(out)


def line(x1, y1, x2, y2, stroke, sw, dash=None, arrow=False, cap=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    a = f' marker-end="url(#{_marker_id(stroke)})"' if arrow else ""
    c = f' stroke-linecap="{cap}"' if cap else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{d}{a}{c}/>'


def arc_arrow(cx, cy, r, start_deg, end_deg, stroke, sw):
    x1 = cx + r * math.cos(math.radians(start_deg))
    y1 = cy + r * math.sin(math.radians(start_deg))
    x2 = cx + r * math.cos(math.radians(end_deg))
    y2 = cy + r * math.sin(math.radians(end_deg))
    large = 1 if abs(end_deg - start_deg) > 180 else 0
    return (f'<path d="M {x1:.2f},{y1:.2f} A {r},{r} 0 {large} 1 {x2:.2f},{y2:.2f}" stroke="{stroke}" '
            f'stroke-width="{sw}" fill="none" marker-end="url(#{_marker_id(stroke)})"/>')


def hand(cx, cy, rot, color, scale=1.0):
    """scale=1.0 == the panel-1 'blue/stabilizer' hand size, expressed relative to FS
    (so it scales automatically with the solved font size)."""
    bw, bh = 1.421 * FS * scale, 1.668 * FS * scale
    fw, fh = 0.340 * FS * scale, 0.926 * FS * scale
    sw = max(0.6, 0.0926 * FS * scale)
    out = [f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({rot})">']
    out.append(rect(-bw / 2, -bh * 0.5, bw, bh, "#FFFFFF", color, sw, rx=0.432 * FS * scale))
    fx0 = -bw / 2 + 0.16 * bw
    gap = fw * 1.26
    for i in range(4):
        out.append(rect(fx0 + i * gap, -bh * 0.5 - fh * 0.74, fw, fh, "#FFFFFF", color, sw * 0.83, rx=0.154 * FS * scale))
    thumb_w, thumb_h = 0.741 * FS * scale, 0.401 * FS * scale
    out.append('<g transform="rotate(-28)">' + rect(-bw / 2 - thumb_w * 0.78, -bh * 0.05, thumb_w, thumb_h, "#FFFFFF", color, sw * 0.83, rx=0.185 * FS * scale) + '</g>')
    out.append('</g>')
    return "\n".join(out)


def bottle(cx, cy, rot, scale=1.0, cap_offset=(0, 0)):
    """scale=1.0 == the panel-2/3 'small' bottle size, relative to FS."""
    bw, bh = 1.150 * FS * scale, 2.250 * FS * scale
    capw, caph = 0.750 * FS * scale, 0.500 * FS * scale
    out = [f'<g transform="translate({cx:.2f},{cy:.2f}) rotate({rot})">']
    out.append(rect(-bw / 2, -bh * 0.39, bw, bh, "#DDDDDD", DARK, max(0.6, 0.0735 * FS * scale), rx=0.25 * FS * scale))
    cap_cx = cap_offset[0] - capw / 2
    cap_cy = -bh * 0.39 + cap_offset[1]
    out.append(rect(cap_cx, cap_cy, capw, caph, "#555555", DARK, max(0.6, 0.0735 * FS * scale), rx=0.125 * FS * scale))
    out.append('</g>')
    return "\n".join(out), (bw, bh)


# ============================= scene assembly =============================
elems = []
elems.append(f'<rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="#FFFFFF"/>')
DEFS_PLACEHOLDER = "<!--DEFS-->"
elems.append(DEFS_PLACEHOLDER)

PANEL_BOTTOM = PANEL_Y + PANEL_H
titles = [TITLE1, TITLE2, TITLE3]
assertions = []


def assert_inside(name, left, right, top, bottom, box_left, box_right, box_top, box_bottom):
    msg = f"{name}: [{left:.1f},{right:.1f}]x[{top:.1f},{bottom:.1f}] vs container [{box_left:.1f},{box_right:.1f}]x[{box_top:.1f},{box_bottom:.1f}]"
    ok = left >= box_left - 1e-6 and right <= box_right + 1e-6 and top >= box_top - 1e-6 and bottom <= box_bottom + 1e-6
    assertions.append((ok, msg))
    if not ok:
        print("OVERFLOW:", msg)


def assert_no_overlap(name_a, a, name_b, b):
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    overlap = ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1
    assertions.append((not overlap, f"{name_a} vs {name_b}: {a} vs {b}"))
    if overlap:
        print("LABEL OVERLAP:", name_a, "vs", name_b)


# ---- panel frames + titles ----
for i in range(3):
    px, pw = panels_x[i], panel_w[i]
    elems.append(rect(px, PANEL_Y, pw, PANEL_H, "#FFFFFF", GRAY, 1.1, rx=FS * 0.3))
    t1, t2 = titles[i]
    b1, b2 = FS * 1.3, FS * 2.8
    elems.append(text(px + pw / 2, b1, t1, FS_TITLE, DARK, True))
    elems.append(text(px + pw / 2, b2, t2, FS_TITLE, DARK, True))
    for t, b in [(t1, b1), (t2, b2)]:
        tw = text_w(t, FS_TITLE, True)
        assert_inside(f"title '{t}'", px + pw / 2 - tw / 2, px + pw / 2 + tw / 2,
                      text_top(b, FS_TITLE), text_bottom(b, FS_TITLE), px, px + pw, 0, PANEL_Y)

# connector triangles between panels
for i in range(2):
    gx0 = panels_x[i] + panel_w[i]
    gx1 = panels_x[i + 1]
    midy = PANEL_Y + PANEL_H / 2
    tri_h = FS * 1.1
    elems.append(f'<path d="M {gx0+FS*0.3:.2f},{midy-tri_h:.2f} L {gx1-FS*0.3:.2f},{midy:.2f} L {gx0+FS*0.3:.2f},{midy+tri_h:.2f} Z" fill="#CCCCCC"/>')

# =========================== PANEL 1: Human ===========================
px, pw = panels_x[0], panel_w[0]
cx = px + pw / 2
lbl_top_y = PANEL_Y + FS * 1.55
lbl_bot_y = PANEL_BOTTOM - FS * 0.75
bottle_cy = PANEL_Y + PANEL_H * 0.56
bottle_scale1 = 1.353
svg_bottle1, (bw1, bh1) = bottle(cx, bottle_cy, 0, scale=bottle_scale1)
elems.append(svg_bottle1)

hand_blue_cx, hand_blue_cy = cx - 1.70 * FS, bottle_cy + 0.41 * FS
elems.append(hand(hand_blue_cx, hand_blue_cy, -18, BLUE, scale=1.0))
cap_cy_abs = bottle_cy - bh1 * 0.39
hand_or_cx, hand_or_cy = cx + 0.55 * FS, cap_cy_abs - 0.15 * FS
elems.append(hand(hand_or_cx, hand_or_cy, 35, ORANGE, scale=0.904))
elems.append(arc_arrow(hand_or_cx + FS * 0.05, hand_or_cy - FS * 0.85, FS * 0.55, -30, 245, ORANGE, FS * 0.1))

elems.append(text(cx, lbl_bot_y, LBL_STAB_HOLD, FS, BLUE, True))
elems.append(text(cx, lbl_top_y, LBL_ACT_TWIST, FS, ORANGE, True))
for s, b, color in [(LBL_STAB_HOLD, lbl_bot_y, BLUE), (LBL_ACT_TWIST, lbl_top_y, ORANGE)]:
    tw = text_w(s, FS, True)
    assert_inside(f"p1 label '{s}'", cx - tw / 2, cx + tw / 2, text_top(b, FS), text_bottom(b, FS),
                  px, px + pw, PANEL_Y, PANEL_BOTTOM)

# =========================== PANEL 2: Symmetric VLA ===========================
px, pw = panels_x[1], panel_w[1]
cx = px + pw / 2
box_w = text_w("VLA policy", FS, False) + 2 * BOX_PAD
box_h = FS * 1.5
box_y = PANEL_Y + FS * 1.3
elems.append(rect(cx - box_w / 2, box_y, box_w, box_h, "#DDDDDD", "#555555", FS * 0.06, rx=FS * 0.25))
elems.append(text(cx, box_y + box_h * 0.68, "VLA policy", FS, DARK, False))

arm_label_y = box_y + box_h + FS * 1.35
arrow_top_y = arm_label_y + FS * 0.45
arrow_bot_y = PANEL_BOTTOM - FS * 2.9
left_arrow_x0, left_arrow_x1 = cx - FS * 0.55, cx - FS * 1.9
right_arrow_x0, right_arrow_x1 = cx + FS * 0.55, cx + FS * 1.9
elems.append(line(left_arrow_x0, arrow_top_y, left_arrow_x1, arrow_bot_y, "#555555", FS * 0.14, arrow=True))
elems.append(line(right_arrow_x0, arrow_top_y, right_arrow_x1, arrow_bot_y, "#555555", FS * 0.14, arrow=True))

left_w = text_w("Left arm", FS, False)
right_w = text_w("Right arm", FS, False)
label_gap = FS * 0.6
left_lbl_x = cx - label_gap / 2 - left_w / 2
right_lbl_x = cx + label_gap / 2 + right_w / 2
elems.append(text(left_lbl_x, arm_label_y, "Left arm", FS, "#555555", False, anchor="middle"))
elems.append(text(right_lbl_x, arm_label_y, "Right arm", FS, "#555555", False, anchor="middle"))
for s, lx, b in [("Left arm", left_lbl_x, arm_label_y), ("Right arm", right_lbl_x, arm_label_y)]:
    tw = text_w(s, FS, False)
    assert_inside(f"p2 label '{s}'", lx - tw / 2, lx + tw / 2, text_top(b, FS), text_bottom(b, FS), px, px + pw, PANEL_Y, PANEL_BOTTOM)
assert_no_overlap("Left arm", (left_lbl_x - left_w / 2, left_lbl_x + left_w / 2, text_top(arm_label_y, FS), text_bottom(arm_label_y, FS)),
                   "Right arm", (right_lbl_x - right_w / 2, right_lbl_x + right_w / 2, text_top(arm_label_y, FS), text_bottom(arm_label_y, FS)))
# labels must sit above the arrows (no vertical overlap with the arrow band)
assert text_bottom(arm_label_y, FS) < arrow_top_y, "arm labels dip into the arrow band"

# failure icon: wavy wobble under left arrow, steady lines under right (symmetric VLA = same treatment, but
# left one wobbles because it happens to be the one holding here) + bottle knocked over + cross mark
wob_cx, wob_cy = left_arrow_x1, arrow_bot_y + FS * 0.55
wob = []
for i in range(21):
    t = i / 20
    wob.append(f"{wob_cx-FS*0.5+t*FS:.2f},{wob_cy+math.sin(t*4*math.pi)*FS*0.18:.2f}")
elems.append(f'<polyline points="{" ".join(wob)}" stroke="{RED}" stroke-width="{FS*0.09}" fill="none"/>')
elems.append(line(wob_cx - FS * 0.22, wob_cy + FS * 0.35, wob_cx - FS * 0.05, wob_cy + FS * 0.95, "#555555", FS * 0.13, cap="round"))
elems.append(line(wob_cx + FS * 0.22, wob_cy + FS * 0.35, wob_cx + FS * 0.05, wob_cy + FS * 0.95, "#555555", FS * 0.13, cap="round"))

fallen_bottle_cx, fallen_bottle_cy = right_arrow_x1 + FS * 0.1, arrow_bot_y + FS * 1.55
svg_bottle2, (bw2, bh2) = bottle(fallen_bottle_cx, fallen_bottle_cy, 25, scale=1.0)
elems.append(svg_bottle2)
elems.append(line(right_arrow_x1 - FS * 0.22, arrow_bot_y + FS * 0.35, right_arrow_x1 - FS * 0.05, arrow_bot_y + FS * 0.95, "#555555", FS * 0.13, cap="round"))
elems.append(line(right_arrow_x1 + FS * 0.22, arrow_bot_y + FS * 0.35, right_arrow_x1 + FS * 0.05, arrow_bot_y + FS * 0.95, "#555555", FS * 0.13, cap="round"))
elems.append(line(right_arrow_x1 - FS * 0.18, arrow_bot_y + FS * 0.08, right_arrow_x1 + FS * 0.18, arrow_bot_y + FS * 0.38, RED, FS * 0.15))
elems.append(line(right_arrow_x1 - FS * 0.18, arrow_bot_y + FS * 0.38, right_arrow_x1 + FS * 0.18, arrow_bot_y + FS * 0.08, RED, FS * 0.15))

# =========================== PANEL 3: ARD-VLA ===========================
px, pw = panels_x[2], panel_w[2]
cx = px + pw / 2
shared_w = text_w("Shared", FS, False) + 2 * BOX_PAD
shared_h = FS * 1.5
shared_y = PANEL_Y + FS * 1.3
elems.append(rect(cx - shared_w / 2, shared_y, shared_w, shared_h, "#DDDDDD", "#555555", FS * 0.06, rx=FS * 0.25))
elems.append(text(cx, shared_y + shared_h * 0.68, "Shared", FS, DARK, False))
for s, b in [("Shared", shared_y + shared_h * 0.68)]:
    tw = text_w(s, FS, False)
    assert_inside(f"p3 '{s}'", cx - tw / 2, cx + tw / 2, text_top(b, FS), text_bottom(b, FS), cx - shared_w / 2, cx + shared_w / 2, shared_y, shared_y + shared_h)

role_y = shared_y + shared_h + FS * 1.55
role_h = FS * 1.4
stab_w = text_w("Stabilizer", FS, False) + 2 * BOX_PAD
act_w = text_w("Actuator", FS, False) + 2 * BOX_PAD
pair_w = stab_w + ROLE_GAP + act_w
pair_x0 = cx - pair_w / 2
stab_x = pair_x0
act_x = pair_x0 + stab_w + ROLE_GAP
assert_inside("role box pair", pair_x0, pair_x0 + pair_w, role_y, role_y + role_h, px, px + pw, PANEL_Y, PANEL_BOTTOM)

elems.append(rect(stab_x, role_y, stab_w, role_h, "#D6E8F5", BLUE, FS * 0.1, rx=FS * 0.22))
elems.append(text(stab_x + stab_w / 2, role_y + role_h * 0.7, "Stabilizer", FS, BLUE, False))
elems.append(rect(act_x, role_y, act_w, role_h, "#FBE6C7", ORANGE, FS * 0.1, rx=FS * 0.22))
elems.append(text(act_x + act_w / 2, role_y + role_h * 0.7, "Actuator", FS, ORANGE, False))
for s, bx, bw_, b in [("Stabilizer", stab_x, stab_w, role_y + role_h * 0.7), ("Actuator", act_x, act_w, role_y + role_h * 0.7)]:
    tw = text_w(s, FS, False)
    cxt = bx + bw_ / 2
    assert_inside(f"p3 '{s}' text", cxt - tw / 2, cxt + tw / 2, text_top(b, FS), text_bottom(b, FS), bx, bx + bw_, role_y, role_y + role_h)

elems.append(line(cx - shared_w * 0.15, shared_y + shared_h, stab_x + stab_w / 2, role_y, "#555555", FS * 0.09))
elems.append(line(cx + shared_w * 0.15, shared_y + shared_h, act_x + act_w / 2, role_y, "#555555", FS * 0.09))

bottle3_cy = role_y + role_h + FS * 3.3
svg_bottle3, (bw3, bh3) = bottle(cx, bottle3_cy, 0, scale=1.0, cap_offset=(0, -FS * 1.1))
# draw a steady arrow from Stabilizer box down toward the bottle, then the gripper, then the
# arrow+checkmark for Actuator, then the bottle (so the gripper sits visually "in front of" the bottle edge)
elems.append(line(stab_x + stab_w / 2, role_y + role_h, stab_x + stab_w / 2, bottle3_cy - bh3 * 0.75, BLUE, FS * 0.12, arrow=True))
elems.append(line(act_x + act_w / 2, role_y + role_h, act_x + act_w / 2, bottle3_cy - bh3 * 0.95, ORANGE, FS * 0.12, arrow=True))
elems.append(arc_arrow(act_x + act_w / 2 + FS * 0.05, bottle3_cy - bh3 * 0.55, FS * 0.5, -50, 225, ORANGE, FS * 0.1))
elems.append(f'<path d="M {act_x+act_w/2-FS*0.35:.2f},{bottle3_cy-bh3*0.45:.2f} L {act_x+act_w/2-FS*0.12:.2f},{bottle3_cy-bh3*0.25:.2f} L {act_x+act_w/2+FS*0.3:.2f},{bottle3_cy-bh3*0.75:.2f}" stroke="{GREEN}" stroke-width="{FS*0.13}" fill="none"/>')
elems.append(svg_bottle3)

# Stabilizer gripper: positioned using the SAME relative hand-to-bottle offset/scale ratio proven
# in panel 1 (offset scaled by the bottle-scale ratio 1.0/1.353), so the fingers visibly overlap
# the bottle body exactly as in panel 1's "grip" composition.
grip_scale = 1.0 / bottle_scale1
grip_cx = cx + (-1.70 * FS) * grip_scale
grip_cy = bottle3_cy + (0.41 * FS) * grip_scale
elems.append(hand(grip_cx, grip_cy, -18, BLUE, scale=grip_scale))

elems = [e if e != DEFS_PLACEHOLDER else f"<defs>{marker_defs()}</defs>" for e in elems]
svg_out = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}" height="{H:.2f}" '
           f'viewBox="0 0 {W:.2f} {H:.2f}">\n' + "\n".join(elems) + "\n</svg>\n")

out_dir = sys.argv[1] if len(sys.argv) > 1 else "paper/figures"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/fig1_teaser.svg", "w") as f:
    f.write(svg_out)

n_fail = sum(1 for ok, _ in assertions if not ok)
print(f"assertions: {len(assertions)} total, {n_fail} failed")

import cairosvg
cairosvg.svg2png(url=f"{out_dir}/fig1_teaser.svg", write_to=f"{out_dir}/fig1_teaser.png", scale=4)
cairosvg.svg2pdf(url=f"{out_dir}/fig1_teaser.svg", write_to=f"{out_dir}/fig1_teaser.pdf")
print("wrote", f"{out_dir}/fig1_teaser.png")

# ---- simplified editable .drawio (boxes/text/arrows only, no procedural icon paths --
#      matches the existing convention for this figure) ----
def mx_rect(id_, x, y, w, h, fill, stroke, sw=1.5, rounded=True):
    r = "rounded=1;" if rounded else ""
    return (f'<mxCell id="{id_}" value="" style="{r}fillColor={fill};strokeColor={stroke};strokeWidth={sw};" '
            f'vertex="1" parent="1"><mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def mx_text(id_, cx, baseline, s, fs, color, bold=True):
    w_ = max(text_w(s, fs, bold) * 1.3, 10)
    x = cx - w_ / 2
    y = baseline - fs
    return (f'<mxCell id="{id_}" value="{s}" style="text;html=1;fontSize={fs:.1f};fontColor={color};align=center;'
            f'fontStyle={1 if bold else 0};" vertex="1" parent="1">'
            f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w_:.1f}" height="{fs*1.4:.1f}" as="geometry"/></mxCell>')


def mx_edge(id_, x1, y1, x2, y2, stroke, sw=2):
    return (f'<mxCell id="{id_}" value="" style="endArrow=block;html=1;strokeColor={stroke};strokeWidth={sw};" '
            f'edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
            f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/><mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
            f'</mxGeometry></mxCell>')


cells = []
n = [0]
def nid():
    n[0] += 1
    return f"n{n[0]}"

for i in range(3):
    pxx, pw_ = panels_x[i], panel_w[i]
    cells.append(mx_rect(nid(), pxx, PANEL_Y, pw_, PANEL_H, "#FFFFFF", "#999999", 1.1))
    t1, t2 = titles[i]
    cells.append(mx_text(nid(), pxx + pw_ / 2, FS * 1.3, t1, FS_TITLE, DARK))
    cells.append(mx_text(nid(), pxx + pw_ / 2, FS * 2.8, t2, FS_TITLE, DARK))

cells.append(mx_text(nid(), panels_x[0] + panel_w[0] / 2, lbl_bot_y, LBL_STAB_HOLD, FS, BLUE))
cells.append(mx_text(nid(), panels_x[0] + panel_w[0] / 2, lbl_top_y, LBL_ACT_TWIST, FS, ORANGE))

px2, pw2 = panels_x[1], panel_w[1]
cx2 = px2 + pw2 / 2
cells.append(mx_rect(nid(), cx2 - box_w / 2, box_y, box_w, box_h, "#DDDDDD", "#555555", 2))
cells.append(mx_text(nid(), cx2, box_y + box_h * 0.68, "VLA policy", FS, DARK, False))
cells.append(mx_text(nid(), left_lbl_x, arm_label_y, "Left arm", FS, "#555555", False))
cells.append(mx_text(nid(), right_lbl_x, arm_label_y, "Right arm", FS, "#555555", False))
cells.append(mx_edge(nid(), left_arrow_x0, arrow_top_y, left_arrow_x1, arrow_bot_y, "#555555", 1.5))
cells.append(mx_edge(nid(), right_arrow_x0, arrow_top_y, right_arrow_x1, arrow_bot_y, "#555555", 1.5))

px3, pw3 = panels_x[2], panel_w[2]
cx3 = px3 + pw3 / 2
cells.append(mx_rect(nid(), cx3 - shared_w / 2, shared_y, shared_w, shared_h, "#DDDDDD", "#555555", 2))
cells.append(mx_text(nid(), cx3, shared_y + shared_h * 0.68, "Shared", FS, DARK, False))
cells.append(mx_rect(nid(), stab_x, role_y, stab_w, role_h, "#D6E8F5", BLUE, 2))
cells.append(mx_text(nid(), stab_x + stab_w / 2, role_y + role_h * 0.7, "Stabilizer", FS, BLUE, False))
cells.append(mx_rect(nid(), act_x, role_y, act_w, role_h, "#FBE6C7", ORANGE, 2))
cells.append(mx_text(nid(), act_x + act_w / 2, role_y + role_h * 0.7, "Actuator", FS, ORANGE, False))
cells.append(mx_edge(nid(), stab_x + stab_w / 2, role_y + role_h, stab_x + stab_w / 2, bottle3_cy, BLUE, 1.5))
cells.append(mx_edge(nid(), act_x + act_w / 2, role_y + role_h, act_x + act_w / 2, bottle3_cy, ORANGE, 1.5))

drawio = f'''<mxfile host="app.diagrams.net">
  <diagram name="Figure 1 - Teaser" id="fig1">
    <mxGraphModel dx="800" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W:.0f}" pageHeight="{H:.0f}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {"".join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
with open(f"{out_dir}/fig1_teaser.drawio", "w") as f:
    f.write(drawio)
print("wrote", f"{out_dir}/fig1_teaser.drawio")
