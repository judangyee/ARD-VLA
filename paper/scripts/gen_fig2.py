#!/usr/bin/env python3
"""Fig.2 architecture diagram generator.

Content/topology is unchanged from the original hand-authored SVG (same boxes,
labels, colors, arrows) -- this only recomputes every y-coordinate from actual
measured text metrics with generous margins, since several boxes (Prefix/Suffix,
the two ARD heads, the Combine rows) were too short for their text at the
original size. x-coordinates and the overall canvas width are left untouched
(that was already exactly 8.0pt at 17.5cm print width; changing it would need
re-deriving the font size).
"""
import os
import sys

from PIL import ImageFont

REG_PATH = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
_font_cache = {}


def _font(fs):
    key = round(fs * 10)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(REG_PATH, max(1, round(fs * 10)))
    return _font_cache[key]


def text_w(s, fs):
    bbox = _font(fs).getbbox(s)
    return (bbox[2] - bbox[0]) / 10.0


FS = 138  # unchanged from the original: 138 * (17.5cm in pt) / W == 8.0pt exactly
W = 8556
PRINT_W_PT = 17.5 * 28.3465
assert abs(FS * PRINT_W_PT / W - 8.0) < 0.05, "canvas width changed; font size is no longer validated at 8pt"

# Conservative, uniform per-line vertical metrics (safe overestimate of the real
# ascent/descent measured empirically off the renderer -- see paper/figures history).
ASCENT, DESCENT, LINE_GAP = 108, 34, 146
PAD_T, PAD_B = 45, 45


def box_h(n_lines):
    return PAD_T + ASCENT + (n_lines - 1) * LINE_GAP + DESCENT + PAD_B


def baselines(top, n_lines):
    first = top + PAD_T + ASCENT
    return [first + i * LINE_GAP for i in range(n_lines)]


BLUE, ORANGE, GRAY, DARK = "#0072B2", "#E69F00", "#555555", "#1A1A1A"

elems = []
assertions = []
mx_cells = []
_mx_n = [0]


def _mxid():
    _mx_n[0] += 1
    return f"n{_mx_n[0]}"


def mx_box(x, y, w, h, lines, fill, stroke, sw=1):
    val = "&#10;".join(lines)
    mx_cells.append(f'<mxCell id="{_mxid()}" value="{val}" style="rounded=1;whiteSpace=wrap;html=1;'
                     f'fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontSize={FS};fontColor={DARK};'
                     f'fontFamily=Helvetica;align=center;verticalAlign=middle;spacing=8;" vertex="1" parent="1">'
                     f'<mxGeometry x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" as="geometry"/></mxCell>')


def mx_label(cx, cy, s, color=DARK):
    w_ = text_w(s, FS) * 1.25 + 40
    mx_cells.append(f'<mxCell id="{_mxid()}" value="{s}" style="text;html=1;fontSize={FS};fontColor={color};'
                     f'align=center;fontFamily=Helvetica;" vertex="1" parent="1">'
                     f'<mxGeometry x="{cx-w_/2:.1f}" y="{cy-ASCENT:.1f}" width="{w_:.1f}" height="{ASCENT+DESCENT:.1f}" as="geometry"/></mxCell>')


def mx_edge(x1, y1, x2, y2, color=DARK, sw=2.2):
    mx_cells.append(f'<mxCell id="{_mxid()}" value="" style="endArrow=block;html=1;strokeColor={color};strokeWidth={sw};" '
                     f'edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                     f'<mxPoint x="{x1:.1f}" y="{y1:.1f}" as="sourcePoint"/><mxPoint x="{x2:.1f}" y="{y2:.1f}" as="targetPoint"/>'
                     f'</mxGeometry></mxCell>')


def assert_inside(name, left, right, top, bottom, bl, br, bt, bb):
    ok = left >= bl - 0.5 and right <= br + 0.5 and top >= bt - 0.5 and bottom <= bb + 0.5
    assertions.append((ok, f"{name}: text/box [{left:.1f},{right:.1f}]x[{top:.1f},{bottom:.1f}] "
                            f"vs container [{bl:.1f},{br:.1f}]x[{bt:.1f},{bb:.1f}]"))
    if not ok:
        print("OVERFLOW:", assertions[-1][1])


def rect(x, y, w, h, fill, stroke, sw=1, rx=6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"{d}/>')


def text(x, y, s, color=DARK, anchor="middle"):
    # DejaVu Sans fallback: Arial/Liberation Sans has no glyph for U+2295 (⊕), used in the Combine rows
    elems.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{FS}" font-family="Arial,Helvetica,\'DejaVu Sans\',sans-serif" '
                 f'fill="{color}" text-anchor="{anchor}">{s}</text>')


def circled_plus_r(cx, cy_line_baseline, label, color):
    """Draws a '(+) <label>' unit (circle+cross icon, since U+2295 is missing from the
    renderer's font fallback chain) centered as a whole at (cx, baseline)."""
    r = ASCENT * 0.42
    icon_cx = cx - (text_w(label, FS) + r * 2.6) / 2 + r
    label_x = icon_cx + r + 18
    icon_cy = cy_line_baseline - ASCENT * 0.38
    elems.append(f'<circle cx="{icon_cx:.1f}" cy="{icon_cy:.1f}" r="{r:.1f}" fill="none" stroke="{color}" stroke-width="5"/>')
    elems.append(f'<line x1="{icon_cx-r*0.6:.1f}" y1="{icon_cy:.1f}" x2="{icon_cx+r*0.6:.1f}" y2="{icon_cy:.1f}" stroke="{color}" stroke-width="5"/>')
    elems.append(f'<line x1="{icon_cx:.1f}" y1="{icon_cy-r*0.6:.1f}" x2="{icon_cx:.1f}" y2="{icon_cy+r*0.6:.1f}" stroke="{color}" stroke-width="5"/>')
    text(label_x + text_w(label, FS) / 2, cy_line_baseline, label, color, anchor="middle")
    return f"⊕ {label}"  # drawio's own renderer has the glyph; no icon workaround needed there


def box_with_text(x, y, w, lines, fill, stroke, sw=1, color=DARK, check_name=None, min_h=0):
    h = max(box_h(len(lines)), min_h)
    rect(x, y, w, h, fill, stroke, sw)
    cx = x + w / 2
    text_block_h = ASCENT + (len(lines) - 1) * LINE_GAP + DESCENT
    first_baseline = y + (h - text_block_h) / 2 + ASCENT
    mx_lines = []
    for i, s in enumerate(lines):
        b = first_baseline + i * LINE_GAP
        if callable(s):
            mx_lines.append(s(cx, b))  # draws the SVG icon, returns the drawio-text equivalent
            continue
        text(cx, b, s, color)
        tw = text_w(s, FS)
        if check_name:
            assert_inside(f"{check_name} '{s}'", cx - tw / 2, cx + tw / 2, b - ASCENT, b + DESCENT, x, x + w, y, y + h)
        mx_lines.append(s)
    mx_box(x, y, w, h, mx_lines, fill, stroke, sw)
    return y + h / 2, y + h  # center, bottom


def label_with_bg(cx, cy, s, color=DARK, pad_x=30, pad_y=16, anchor="middle"):
    tw = text_w(s, FS)
    bg_w, bg_h = tw + 2 * pad_x, ASCENT + DESCENT + 2 * pad_y
    bg_x = cx - bg_w / 2 if anchor == "middle" else (cx if anchor == "start" else cx - bg_w)
    elems.append(f'<rect x="{bg_x:.1f}" y="{cy-ASCENT-pad_y:.1f}" width="{bg_w:.1f}" height="{bg_h:.1f}" '
                 f'fill="#FFFFFF" opacity="0.85"/>')
    text(cx, cy + (ASCENT - DESCENT) / 2, s, color, anchor=anchor)
    mx_cx = bg_x + bg_w / 2 if anchor != "middle" else cx
    mx_label(mx_cx, cy, s, color)


_marker_ids = {}


def marker(color):
    if color not in _marker_ids:
        _marker_ids[color] = f"arrow{len(_marker_ids)}"
    return _marker_ids[color]


def polyline(pts, color=DARK, sw=2.2):
    mid = marker(color)
    pstr = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    elems.append(f'<polyline points="{pstr}" fill="none" stroke="{color}" stroke-width="{sw}" marker-end="url(#{mid})"/>')
    mx_edge(*pts[0], *pts[-1], color, sw)


# ============================= COLUMN 1: inputs =============================
IN_X, IN_W = 20.0, 1015.6
in_labels = [["Camera", "image(s)"], ["Language", "instruction"], ["Robot", "state"], ["Noisy action", "x_t, time t"]]
in_y = 370.0
in_centers = []
for lines in in_labels:
    c, bot = box_with_text(IN_X, in_y, IN_W, lines, "#FFFFFF", DARK, check_name="input")
    in_centers.append(c)
    in_y = bot + 22

# ============================= COLUMN 2: SmolVLM2 / Prefix+Suffix =============================
SM_X, SM_W = 1077.6, 1308.0
SM_TOP = 360.0
title_b = baselines(SM_TOP, 3)  # "SmolVLM2 +" / "Action-Expert" / "(interleaved)" -- unchanged position
cx2 = SM_X + SM_W / 2
text(cx2, title_b[0], "SmolVLM2 +")
text(cx2, title_b[1], "Action-Expert")
text(cx2, title_b[2], "(interleaved)")
for s, b in zip(["SmolVLM2 +", "Action-Expert", "(interleaved)"], title_b):
    tw = text_w(s, FS)
    assert_inside(f"SmolVLM2 title '{s}'", cx2 - tw / 2, cx2 + tw / 2, b - ASCENT, b + DESCENT, SM_X, SM_X + SM_W, SM_TOP, SM_TOP + 10_000)
    mx_label(cx2, b, s)

PS_X, PS_W = 1389.4, 684.4
prefix_top = title_b[2] + DESCENT + 40
prefix_c, prefix_bot = box_with_text(PS_X, prefix_top, PS_W, ["Prefix", "(VLM)"], "#EDEDED", DARK, check_name="Prefix")
suffix_top = prefix_bot + 240
suffix_c, suffix_bot = box_with_text(PS_X, suffix_top, PS_W, ["Suffix", "(expert)"], "#EDEDED", DARK, check_name="Suffix")
SM_BOT = suffix_bot + 40
rect(SM_X, SM_TOP, SM_W, SM_BOT - SM_TOP, "none", DARK, 1)
mx_box(SM_X, SM_TOP, SM_W, SM_BOT - SM_TOP, [], "none", DARK, 1)
assert_inside("Prefix box", PS_X, PS_X + PS_W, prefix_top, prefix_bot, SM_X, SM_X + SM_W, SM_TOP, SM_BOT)
assert_inside("Suffix box", PS_X, PS_X + PS_W, suffix_top, suffix_bot, SM_X, SM_X + SM_W, SM_TOP, SM_BOT)

# input -> Prefix/Suffix arrows
for i in range(3):  # camera, language, robot-state -> Prefix
    polyline([(IN_X + IN_W, in_centers[i]), (IN_X + IN_W + 21, in_centers[i]), (IN_X + IN_W + 21, prefix_c), (PS_X, prefix_c)])
polyline([(IN_X + IN_W, in_centers[3]), (IN_X + IN_W + 21, in_centers[3]), (IN_X + IN_W + 21, suffix_c), (PS_X, suffix_c)])

# Prefix -> Suffix cross-attention arrow + label (placed in the empty gap between the boxes)
polyline([(PS_X + PS_W / 2, prefix_bot), (PS_X + PS_W / 2, suffix_top)])
gap_mid = (prefix_bot + suffix_top) / 2
label_with_bg(PS_X + PS_W / 2 + 362, gap_mid, "cross-attention")

# ============================= COLUMN 3: W_out =============================
WO_X, WO_W = 3035.6, 850.0
MID_Y = (prefix_c + suffix_c) / 2  # shared vertical center for the right-hand columns
wo_top = MID_Y - box_h(4) / 2
wo_c, wo_bot = box_with_text(WO_X, wo_top, WO_W, ["W_out", "Linear", "720→32", "v_t (base)"], "#EDEDED", DARK, check_name="W_out")

# suffix_out fan-out (Suffix -> W_out, Stabilizer head, Actuator head)
SO_X = PS_X + PS_W
SO_BEND_X = SO_X + 325
HD_X, HD_W = 3927.6, 850.0
HD_GAP = 40
hd_pair_h = 2 * box_h(6) + HD_GAP
stab_top = MID_Y - hd_pair_h / 2
act_top = stab_top + box_h(6) + HD_GAP
stab_c = stab_top + box_h(6) / 2
act_c = act_top + box_h(6) / 2

polyline([(SO_X, suffix_c), (SO_BEND_X, suffix_c), (SO_BEND_X, wo_c), (WO_X, wo_c)])
polyline([(SO_X, suffix_c), (SO_BEND_X, suffix_c), (SO_BEND_X, stab_c), (HD_X, stab_c)])
polyline([(SO_X, suffix_c), (SO_BEND_X, suffix_c), (SO_BEND_X, act_c), (HD_X, act_c)])
label_with_bg(SO_BEND_X, suffix_c + 110, "suffix_out")

# ============================= COLUMN 4: Stabilizer / Actuator heads =============================
stab_c2, stab_bot = box_with_text(HD_X, stab_top, HD_W, ["Stabilizer", "head", "720→360", "GELU", "360→8", "(0-init)"],
                                   "#D6E8F5", BLUE, sw=4, check_name="Stabilizer head")
act_c2, act_bot = box_with_text(HD_X, act_top, HD_W, ["Actuator", "head", "720→360", "GELU", "360→8", "(0-init)"],
                                 "#FBE6C7", ORANGE, sw=4, check_name="Actuator head")
assert abs(stab_c2 - stab_c) < 0.1 and abs(act_c2 - act_c) < 0.1

# ============================= COLUMN 5: rho role->arm =============================
RHO_X, RHO_W = 5187.6, 850.0
rho_top = MID_Y - box_h(4) / 2
rho_c, rho_bot = box_with_text(RHO_X, rho_top, RHO_W, ["ρ role→arm", "(config)", "left=Stab", "right=Act"],
                                "#FFFFFF", DARK, sw=4, check_name="rho")
assert abs(rho_c - wo_c) < 0.1  # same shared vertical center as W_out, for a clean column alignment

ENTRY_OFFSET = 150
rho_in_stab_y = rho_c - ENTRY_OFFSET
rho_in_act_y = rho_c + ENTRY_OFFSET

polyline([(HD_X + HD_W, stab_c), (HD_X + HD_W + 205, stab_c), (HD_X + HD_W + 205, rho_in_stab_y), (RHO_X, rho_in_stab_y)], color=DARK)
label_with_bg(HD_X + HD_W + 248, (stab_c + rho_in_stab_y) / 2 - 15, "h_stab")
polyline([(HD_X + HD_W, act_c), (HD_X + HD_W + 205, act_c), (HD_X + HD_W + 205, rho_in_act_y), (RHO_X, rho_in_act_y)], color=DARK)
label_with_bg(HD_X + HD_W + 207, (act_c + rho_in_act_y) / 2 + 15, "h_act")

# ============================= COLUMN 6: Combine =============================
CB_X, CB_W = 6281.6, 850.0
cb_gap = 24
rows_h = 3 * box_h(3) + 2 * cb_gap
row1_top = MID_Y - rows_h / 2
row1_c, row1_bot = box_with_text(CB_X, row1_top, CB_W, ["v_t[0:8]", lambda cx_, b: circled_plus_r(cx_, b, "r_L", BLUE), "→ left"],
                                  "#D6E8F5", BLUE, sw=2, check_name="combine row1")
row2_top = row1_bot + cb_gap
row2_c, row2_bot = box_with_text(CB_X, row2_top, CB_W, ["v_t[8:16]", lambda cx_, b: circled_plus_r(cx_, b, "r_R", ORANGE), "→ right"],
                                  "#FBE6C7", ORANGE, sw=2, check_name="combine row2")
row3_top = row2_bot + cb_gap
row3_c, row3_bot = box_with_text(CB_X, row3_top, CB_W, ["v_t[16:32]", "unchanged"], "#EDEDED", GRAY, sw=1,
                                  check_name="combine row3", min_h=box_h(3))

CMB_X, CMB_W = CB_X - 14, CB_W + 28
combine_title_h = 215
CMB_TOP = row1_top - combine_title_h
CMB_BOT = row3_bot + 40
rect(CMB_X, CMB_TOP, CMB_W, CMB_BOT - CMB_TOP, "none", DARK, 4)
mx_box(CMB_X, CMB_TOP, CMB_W, CMB_BOT - CMB_TOP, [], "none", DARK, 4)
cmb_title_baseline = CMB_TOP + PAD_T + ASCENT
text(CMB_X + CMB_W / 2, cmb_title_baseline, "Combine")
assert_inside("Combine title", CMB_X + CMB_W / 2 - text_w("Combine", FS) / 2, CMB_X + CMB_W / 2 + text_w("Combine", FS) / 2,
              cmb_title_baseline - ASCENT, cmb_title_baseline + DESCENT, CMB_X, CMB_X + CMB_W, CMB_TOP, row1_top)
mx_label(CMB_X + CMB_W / 2, cmb_title_baseline, "Combine")

polyline([(RHO_X + RHO_W, rho_in_stab_y), (RHO_X + RHO_W + 115, rho_in_stab_y), (RHO_X + RHO_W + 115, row1_c), (CB_X, row1_c)], color=BLUE)
label_with_bg(RHO_X + RHO_W + 124, (rho_in_stab_y + row1_c) / 2 - 10, "r_L", color=BLUE, pad_x=14)
polyline([(RHO_X + RHO_W, rho_in_act_y), (RHO_X + RHO_W + 115, rho_in_act_y), (RHO_X + RHO_W + 115, row2_c), (CB_X, row2_c)], color=ORANGE)
label_with_bg(RHO_X + RHO_W + 124, (rho_in_act_y + row2_c) / 2 + 10, "r_R", color=ORANGE, pad_x=14)

# ============================= top feedback arrow: v_t all rows -> Combine =============================
TOP_Y = 20.0
polyline([(WO_X + WO_W, wo_c), (WO_X + WO_W, TOP_Y), (CMB_X + CMB_W / 2, TOP_Y), (CMB_X + CMB_W / 2, CMB_TOP)])
label_with_bg((WO_X + WO_W + CMB_X + CMB_W / 2) / 2, TOP_Y + 95, "v_t (B×T×32) → all rows")

# ============================= COLUMN 7: outputs =============================
CMB_RIGHT = CMB_X + CMB_W
out_label_w = max(text_w(s, FS) for s in ["v_t", "(left)", "(right)", "(unused)"])
arrow1_len = 90
out_x = CMB_RIGHT + arrow1_len + out_label_w / 2 + 15
label_with_bg(out_x, row1_c - 73, "v_t")
label_with_bg(out_x, row1_c + 73, "(left)")
label_with_bg(out_x, row2_c - 73, "v_t")
label_with_bg(out_x, row2_c + 73, "(right)")
label_with_bg(out_x, row3_c + (ASCENT - DESCENT) / 2, "(unused)")
for s, cx_, cy_ in [("v_t", out_x, row1_c - 73), ("(left)", out_x, row1_c + 73), ("v_t", out_x, row2_c - 73),
                     ("(right)", out_x, row2_c + 73), ("(unused)", out_x, row3_c + (ASCENT - DESCENT) / 2)]:
    tw = text_w(s, FS)
    assert_inside(f"output label '{s}'", cx_ - tw / 2, cx_ + tw / 2, cy_ - ASCENT, cy_ + DESCENT, CMB_RIGHT, W, 0, H if "H" in dir() else 1e9)

out_arrow_end = out_x - out_label_w / 2 - 15
for rc in (row1_c, row2_c, row3_c):
    polyline([(CMB_RIGHT, rc), (out_arrow_end, rc)])

merge_gap = 90
merge_start = out_x + out_label_w / 2 + 15
merge_x = merge_start + merge_gap
bracket_x = merge_x + 20
polyline([(merge_start, row1_c), (merge_x, row1_c), (merge_x, (row1_c + row2_c) / 2), (merge_x + 20, (row1_c + row2_c) / 2)])
polyline([(merge_start, row2_c), (merge_x, row2_c), (merge_x, (row1_c + row2_c) / 2), (merge_x + 20, (row1_c + row2_c) / 2)])
bracket_lines = ["N-step", "denoise", "→ chunk"]
bracket_c = (row1_c + row2_c) / 2
bracket_w = max(text_w(s, FS) for s in bracket_lines)
for s, b in zip(bracket_lines, baselines(bracket_c - 1.5 * LINE_GAP, 3)):
    text(bracket_x + bracket_w / 2, b, s)
    mx_label(bracket_x + bracket_w / 2, b, s)
assert bracket_x + bracket_w <= W - 40, f"bracket text runs off canvas: {bracket_x + bracket_w:.1f} > {W - 40}"

# ============================= legend =============================
LEG_Y = CMB_BOT + 190
elems.append(f'<line x1="0" y1="{LEG_Y-80:.1f}" x2="{W}" y2="{LEG_Y-80:.1f}" stroke="#CCCCCC" stroke-width="1.5"/>')
legend = [("Stabilizer (left arm)", BLUE, 20.0), ("Actuator (right arm)", ORANGE, 1982.8), ("Shared / base (v_t, W_out)", GRAY, 3862.8)]
for label_, color, x0 in legend:
    rect(x0, LEG_Y, 156, 156, color, color, 0, rx=0)
    mx_box(x0, LEG_Y, 156, 156, [], color, color, 0)
    text(x0 + 166, LEG_Y + 124.8, label_, anchor="start")
    mx_label(x0 + 166 + text_w(label_, FS) / 2, LEG_Y + 124.8, label_)

H = LEG_Y + 156 + 20

# ============================= write out =============================
defs = "".join(f'<marker id="{mid}" markerWidth="10" markerHeight="10" refX="8" refY="4" orient="auto">'
               f'<path d="M0,0 L8,4 L0,8 Z" fill="{color}"/></marker>' for color, mid in _marker_ids.items())
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H:.0f}" viewBox="0 0 {W} {H:.0f}">\n'
       f'<rect x="0" y="0" width="{W}" height="{H:.0f}" fill="#FFFFFF"/>\n<defs>{defs}</defs>\n'
       + "\n".join(elems) + "\n</svg>\n")

out_dir = sys.argv[1] if len(sys.argv) > 1 else "paper/figures"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/fig2_architecture.svg", "w") as f:
    f.write(svg)

n_fail = sum(1 for ok, _ in assertions if not ok)
print(f"FS={FS} W={W} H={H:.0f} finalpt={FS*PRINT_W_PT/W:.3f}")
print(f"assertions: {len(assertions)} total, {n_fail} failed")

import cairosvg
cairosvg.svg2png(url=f"{out_dir}/fig2_architecture.svg", write_to=f"{out_dir}/fig2_architecture.png", scale=1)
cairosvg.svg2pdf(url=f"{out_dir}/fig2_architecture.svg", write_to=f"{out_dir}/fig2_architecture.pdf")
print("wrote", f"{out_dir}/fig2_architecture.png")

drawio = f'''<mxfile host="app.diagrams.net">
  <diagram name="Figure 2 - ARD-VLA Architecture" id="fig2">
    <mxGraphModel dx="800" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{W}" pageHeight="{H:.0f}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {"".join(mx_cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
with open(f"{out_dir}/fig2_architecture.drawio", "w") as f:
    f.write(drawio)
print("wrote", f"{out_dir}/fig2_architecture.drawio")
