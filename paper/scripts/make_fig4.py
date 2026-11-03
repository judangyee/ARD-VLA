#!/usr/bin/env python3
"""Fig.4 (2-panel 분석 그림) 생성 스크립트.

입력: `RolloutLogger`(`lerobot.policies.smolvla.eval_logging`)가 저장한 npz 로그들
(평가 rollout마다 1개, 메서드별로 여러 개). 출력: 2패널 PDF/SVG(벡터) + 확인용 PNG.

  (a) 접촉 구간(contact>0.5)의 Stabilizer 물체 흔들림 RMS / Actuator jerk RMS 분포.
      SmolVLA / 대칭 이중 헤드 / ARD-VLA를 x축 위치 + 해치 패턴으로 구분(색 아님).
  (b) ARD-VLA 로그의 작업 단계(stage)별 ||r_stab||, ||r_act|| 평균 ± 표준편차
      (Stabilizer=파랑 #0072B2, Actuator=주황 #E69F00 — 공통 색 규칙).

실제 평가 로그가 없으면 그냥 빈/부분 그림이 된다 — 숫자를 지어내지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party/lerobot/src"))
from lerobot.policies.smolvla.eval_logging import RolloutLogger  # noqa: E402

STABILIZER_COLOR = "#0072B2"
ACTUATOR_COLOR = "#E69F00"
METHOD_ORDER = ["smolvla", "symmetric_dual_head", "ard_vla"]
METHOD_LABELS = {"smolvla": "SmolVLA", "symmetric_dual_head": "Symmetric\ndual-head", "ard_vla": "ARD-VLA"}
METHOD_HATCHES = {"smolvla": "", "symmetric_dual_head": "///", "ard_vla": "xxx"}


def _rms(diffs: np.ndarray) -> float:
    """diffs: (N, D) step-to-step vectors -> RMS of their magnitude."""
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=-1))))


def compute_contact_metrics(log: dict) -> tuple[float | None, float | None, str]:
    """접촉 구간의 (stabilizer wobble RMS, actuator jerk RMS)를 계산.
    필요한 키가 없거나 접촉 구간이 너무 짧으면 (None, None, 이유)를 반환한다 —
    숫자를 지어내지 않는다. 로그에 스텝 간 dt가 없어서 전부 스텝당 유한차분(raw finite
    difference, 단위 m)이다 — "jerk"도 실제 m/s^3가 아니라 3차 스텝차분."""
    if "contact" not in log:
        return None, None, "no 'contact' key in log"
    contact_idx = np.where(np.asarray(log["contact"]) > 0.5)[0]
    if len(contact_idx) < 4:
        return None, None, f"contact window too short ({len(contact_idx)} steps)"

    wobble_rms = None
    if "stabilizer_object_pos" in log:
        obj_pos = np.asarray(log["stabilizer_object_pos"])[contact_idx]
        wobble_rms = _rms(np.diff(obj_pos, axis=0))

    jerk_rms = None
    if "ee_pos" in log:
        ee_pos = np.asarray(log["ee_pos"])[contact_idx]
        # ponytail: ee_pos가 (T, 2, D)면 역할 매핑 기본값(오른팔=Actuator)대로 index 1을 쓴다.
        # (T, D) 단일 팔 로그면 그게 이미 Actuator EE라는 가정 — 실제 ARD-Gen observe() 스키마가
        # 확정되면 여기만 고치면 됨.
        actuator_ee = ee_pos[:, 1] if ee_pos.ndim == 3 else ee_pos
        jerk = np.diff(actuator_ee, n=3, axis=0)
        if len(jerk) >= 1:
            jerk_rms = _rms(jerk)

    return wobble_rms, jerk_rms, ""


def compute_stage_role_norms(log: dict) -> dict[str, tuple[float, float, float, float, int]]:
    """stage -> (r_stab_mean, r_stab_std, r_act_mean, r_act_std, n). r_*_norm이 없는
    로그(ARD가 아닌 베이스라인)는 빈 dict를 반환한다."""
    if "r_stab_norm" not in log or "r_act_norm" not in log:
        return {}
    stages = np.asarray(log["stage"])
    r_stab, r_act = np.asarray(log["r_stab_norm"]), np.asarray(log["r_act_norm"])
    out = {}
    for stage in dict.fromkeys(stages.tolist()):  # 등장 순서 보존한 unique
        mask = stages == stage
        vals_stab, vals_act = r_stab[mask], r_act[mask]
        vals_stab, vals_act = vals_stab[~np.isnan(vals_stab)], vals_act[~np.isnan(vals_act)]
        if len(vals_stab) == 0:
            continue
        out[str(stage)] = (
            float(np.mean(vals_stab)), float(np.std(vals_stab)),
            float(np.mean(vals_act)), float(np.std(vals_act)), len(vals_stab),
        )
    return out


def _plot_panel_a(ax, wobble_by_method: dict, jerk_by_method: dict) -> None:
    positions, labels, data, hatches = [], [], [], []
    x = 0.0
    group_centers = []
    for group_name, by_method in [("Stabilizer\nwobble RMS", wobble_by_method), ("Actuator\njerk RMS", jerk_by_method)]:
        group_start = x
        for method in METHOD_ORDER:
            vals = by_method.get(method, [])
            if not vals:
                continue
            positions.append(x)
            data.append(vals)
            labels.append(METHOD_LABELS[method])
            hatches.append(METHOD_HATCHES[method])
            x += 1.0
        group_centers.append(((group_start + x - 1.0) / 2.0, group_name))
        x += 2.0  # group gap

    if not data:
        ax.text(0.5, 0.5, "no contact-window data in logs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    for patch, hatch in zip(bp["boxes"], hatches):
        patch.set_facecolor("white")
        patch.set_edgecolor("black")
        patch.set_hatch(hatch)
    for median in bp["medians"]:
        median.set_color("black")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ymax = max(max(v) for v in data) * 1.35
    ax.set_ylim(top=ymax)
    for center_x, group_name in group_centers:
        ax.text(center_x, ymax * 0.93, group_name, ha="center", va="top", fontsize=9)
    ax.set_ylabel("RMS (m, finite-diff)", fontsize=9)


def _plot_panel_b(ax, stage_norms: dict) -> None:
    if not stage_norms:
        ax.text(0.5, 0.5, "no ARD-VLA role-split logs", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    stages = list(stage_norms.keys())
    x = np.arange(len(stages))
    stab_mean = [stage_norms[s][0] for s in stages]
    stab_std = [stage_norms[s][1] for s in stages]
    act_mean = [stage_norms[s][2] for s in stages]
    act_std = [stage_norms[s][3] for s in stages]

    width = 0.35
    ax.bar(x - width / 2, stab_mean, width, yerr=stab_std, capsize=3, color=STABILIZER_COLOR, label="||r_stab||")
    ax.bar(x + width / 2, act_mean, width, yerr=act_std, capsize=3, color=ACTUATOR_COLOR, label="||r_act||")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("residual norm", fontsize=9)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="RolloutLogger npz 로그 파일(들) 또는 글롭 패턴")
    parser.add_argument("--out", default="paper/figures/fig4_analysis", help="출력 경로 접두사(확장자 없이)")
    parser.add_argument("--watermark", default=None, help="있으면 대각선 워터마크 텍스트를 그림 전체에 찍음(e.g. 'SMOKE TEST')")
    args = parser.parse_args()

    log_paths: list[str] = []
    for pattern in args.logs:
        matched = sorted(str(p) for p in Path().glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
        log_paths.extend(matched)
    if not log_paths:
        raise SystemExit(f"no log files matched: {args.logs}")

    wobble_by_method: dict[str, list[float]] = {m: [] for m in METHOD_ORDER}
    jerk_by_method: dict[str, list[float]] = {m: [] for m in METHOD_ORDER}
    stage_norms: dict[str, tuple] = {}
    skipped = []

    for path in log_paths:
        log = RolloutLogger.load(path)
        method = str(log["method"])
        wobble_rms, jerk_rms, reason = compute_contact_metrics(log)
        if wobble_rms is not None and method in wobble_by_method:
            wobble_by_method[method].append(wobble_rms)
        if jerk_rms is not None and method in jerk_by_method:
            jerk_by_method[method].append(jerk_rms)
        if reason:
            skipped.append(f"{path}: {reason}")

        if method == "ard_vla":
            for stage, vals in compute_stage_role_norms(log).items():
                # 여러 로그에 걸친 같은 stage는 단순 평균(ponytail: 가중 안 함, 로그 적을 때는 영향 없음)
                if stage in stage_norms:
                    prev = stage_norms[stage]
                    stage_norms[stage] = tuple((a + b) / 2 for a, b in zip(prev[:4], vals[:4])) + (prev[4] + vals[4],)
                else:
                    stage_norms[stage] = vals

    for msg in skipped:
        print(f"[make_fig4] skip: {msg}", file=sys.stderr)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    _plot_panel_a(axes[0], wobble_by_method, jerk_by_method)
    _plot_panel_b(axes[1], stage_norms)
    axes[0].set_title("(a)", fontsize=10, loc="left")
    axes[1].set_title("(b)", fontsize=10, loc="left")
    fig.tight_layout()

    if args.watermark:
        fig.text(0.5, 0.5, args.watermark, fontsize=40, color="red", alpha=0.3,
                  ha="center", va="center", rotation=30, zorder=100)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"[make_fig4] wrote {out}.pdf / .svg / .png")


if __name__ == "__main__":
    main()
