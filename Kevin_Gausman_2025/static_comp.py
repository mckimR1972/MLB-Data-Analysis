"""
슬라이더 피안타율 원인 분석
═══════════════════════════════
가설 검증:
  1) 구속 차이 — FF/FS 대비 SL의 속도 갭은 얼마나 되는가?
  2) 무브먼트 중복 — SL vs FS 수직 낙차가 얼마나 겹치는가?
  3) 수평 변화 부족 — SL이 배트 경로를 벗어나지 못하는가?
  4) 로케이션 문제 — 맞힌 SL은 어디에 위치했는가?
  5) 타구 품질 — SL 피안타의 Exit Velo / Launch Angle 분포

좌타/우타 분할 × 6패널 Figure
"""

import oracledb as DB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import Optional

# ══════════════════════════════════════════════════════════
# 공통 설정
# ══════════════════════════════════════════════════════════

DEFAULT_PITCH_COLORS = {
    "FF": "#D83E23", "SI": "#7F77DD", "FC": "#d4a017",
    "SL": "#EC7DB1", "SV": "#d03030", "ST": "#c02828",
    "CU": "#40b040", "KC": "#38a038", "CH": "#4090d0",
    "FS": "#1DD1D1", "KN": "#8060b0",
}

DEFAULT_PITCH_LABELS = {
    "FF": "4-Seam",  "SI": "Sinker",   "FC": "Cutter",
    "SL": "Slider",  "SV": "Sweeper",  "ST": "Sw.Curve",
    "CU": "Curve",   "KC": "Kn.Curve", "CH": "Change",
    "FS": "Splitter", "KN": "Knuckle",
}

SZ_LEFT, SZ_RIGHT = -0.833, 0.833
SZ_BOT, SZ_TOP = 1.5, 3.5


def draw_strike_zone(ax, color="#999", lw=1.5):
    zone = Rectangle((SZ_LEFT, SZ_BOT), SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOT,
                       fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(zone)


# ══════════════════════════════════════════════════════════
# 전처리
# ══════════════════════════════════════════════════════════

def prep_slider_analysis(df):
    df = df.copy()

    hit_events = ["single", "double", "triple", "home_run"]
    non_ab_events = [
        "walk", "hit_by_pitch", "sac_bunt", "sac_fly",
        "sac_bunt_double_play", "sac_fly_double_play",
        "catcher_interf", "intent_walk",
    ]
    whiff_desc = ["swinging_strike", "swinging_strike_blocked"]
    swing_desc = whiff_desc + [
        "foul", "foul_tip", "foul_bunt", "bunt_foul_tip",
        "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
    ]
    inplay_desc = ["hit_into_play", "hit_into_play_no_out", "hit_into_play_score"]

    df["HAS_EVENT"] = df["PITCH_EVENTS"].notna() & (df["PITCH_EVENTS"] != "")
    df["IS_HIT"] = np.where(df["PITCH_EVENTS"].isin(hit_events), 1, 0)
    df["IS_AB"] = np.where(df["HAS_EVENT"] & ~df["PITCH_EVENTS"].isin(non_ab_events), 1, 0)
    df["IS_WHIFF"] = np.where(df["PITCH_DESCRIPTION"].isin(whiff_desc), 1, 0)
    df["IS_SWING"] = np.where(df["PITCH_DESCRIPTION"].isin(swing_desc), 1, 0)
    df["IS_INPLAY"] = np.where(df["PITCH_DESCRIPTION"].isin(inplay_desc), 1, 0)

    # 존 내/밖 분류
    df["IN_ZONE"] = (
        (df["PLATE_X"] >= SZ_LEFT) & (df["PLATE_X"] <= SZ_RIGHT) &
        (df["PLATE_Z"] >= SZ_BOT) & (df["PLATE_Z"] <= SZ_TOP)
    ).astype(int)

    return df


# ══════════════════════════════════════════════════════════
# 메인 시각화 (6패널)
# ══════════════════════════════════════════════════════════

def plot_slider_diagnosis(df, pitcher_name, stand_label, pitch_colors,
                          save_path=None, dpi=150):
    """
    6패널:
      [0,0] 구종별 구속 분포 비교 (FF vs FS vs SL)
      [0,1] 구속 갭 & Whiff% 비교 테이블
      [1,0] PFX_X vs PFX_Z — 무브먼트 클러스터 (FF/FS/SL)
      [1,1] 수직 낙차(PFX_Z) 분포 비교 — SL vs FS 겹침 확인
      [2,0] SL 로케이션 — 피안타 vs 아웃
      [2,1] SL 피안타 타구 품질 (Exit Velo vs Launch Angle)
    """
    # 주요 구종만 추출
    target_types = ["FF", "FS", "SL"]
    available = [pt for pt in target_types if pt in df["PITCH_TYPE"].values]
    if "SL" not in available:
        print(f"[SKIP] No slider data for {stand_label}")
        return None

    fig, axes = plt.subplots(3, 2, figsize=(18, 22))
    fig.patch.set_facecolor("white")
    fig.suptitle(f"{pitcher_name}  —  {stand_label}\n슬라이더 피안타율 원인 분석",
                 fontsize=20, fontweight="bold", y=0.98, color="#222")

    sub = df[df["PITCH_TYPE"].isin(available)].copy()
    sl_df = df[df["PITCH_TYPE"] == "SL"].copy()

    # ════════════════════════════════════════════
    # [0,0] 구종별 구속 분포 (바이올린)
    # ════════════════════════════════════════════
    ax = axes[0, 0]
    velo_data = []
    velo_labels = []
    velo_colors = []
    for pt in available:
        v = df[df["PITCH_TYPE"] == pt]["RELEASE_SPEED"].dropna()
        if len(v) > 0:
            velo_data.append(v)
            velo_labels.append(f"{DEFAULT_PITCH_LABELS.get(pt, pt)}\n(n={len(v)})")
            velo_colors.append(pitch_colors.get(pt, "#999"))

    parts = ax.violinplot(velo_data, positions=range(len(velo_data)),
                          showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(velo_colors[i])
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("white")

    for i, (pt, data) in enumerate(zip(available, velo_data)):
        avg = data.mean()
        ax.text(i, data.max() + 0.5, f"avg {avg:.1f}",
                ha="center", fontsize=10, fontweight="bold", color=velo_colors[i])

    # 구속 갭 표시
    if "FF" in available and "SL" in available:
        ff_avg = df[df["PITCH_TYPE"] == "FF"]["RELEASE_SPEED"].mean()
        sl_avg = df[df["PITCH_TYPE"] == "SL"]["RELEASE_SPEED"].mean()
        gap = ff_avg - sl_avg
        ax.annotate(f"Gap: {gap:.1f} mph",
                    xy=(0.5, (ff_avg + sl_avg) / 2),
                    fontsize=12, fontweight="bold", color="#C62828",
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3E0", edgecolor="#C62828"))

    ax.set_xticks(range(len(velo_data)))
    ax.set_xticklabels(velo_labels, fontsize=11)
    ax.set_ylabel("Velocity (mph)", fontsize=11)
    ax.set_title("Velocity Distribution For Each Pitch Type", fontsize=14, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # ════════════════════════════════════════════
    # [0,1] 구종별 핵심 지표 비교 테이블
    # ════════════════════════════════════════════
    ax = axes[0, 1]
    ax.axis("off")

    table_data = []
    headers = ["P.T", "CNT", "V Avg", "V Max", "PFX_X", "PFX_Z",
               "Whiff%", "AB", "H", "AVG"]

    for pt in available:
        p = df[df["PITCH_TYPE"] == pt]
        n = len(p)
        velo_avg = p["RELEASE_SPEED"].mean()
        velo_max = p["RELEASE_SPEED"].max()
        pfx_x = p["PFX_X"].mean()
        pfx_z = p["PFX_Z"].mean()
        swings = p["IS_SWING"].sum()
        whiff = p["IS_WHIFF"].sum() / swings * 100 if swings > 0 else 0
        ab = p["IS_AB"].sum()
        hits = p["IS_HIT"].sum()
        avg = hits / ab if ab > 0 else 0

        table_data.append([
            DEFAULT_PITCH_LABELS.get(pt, pt),
            f"{n}", f"{velo_avg:.1f}", f"{velo_max:.1f}",
            f"{pfx_x:.2f}", f"{pfx_z:.2f}",
            f"{whiff:.1f}%",
            f"{ab}", f"{hits}", f".{int(avg * 1000):03d}",
        ])

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # 헤더 스타일
    for j in range(len(headers)):
        table[0, j].set_facecolor("#333")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # 행 색상
    for i, pt in enumerate(available):
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(pitch_colors.get(pt, "#eee"))
            table[i + 1, j].set_alpha(0.2)

    ax.set_title("Key Indicator Comparison Of Each Pitch Type", fontsize=14, fontweight="bold", pad=20)

    # ════════════════════════════════════════════
    # [1,0] PFX_X vs PFX_Z 무브먼트 클러스터
    # ════════════════════════════════════════════
    ax = axes[1, 0]
    for pt in available:
        p = df[df["PITCH_TYPE"] == pt].dropna(subset=["PFX_X", "PFX_Z"])
        ax.scatter(p["PFX_X"], p["PFX_Z"],
                   c=pitch_colors.get(pt, "#999"),
                   label=DEFAULT_PITCH_LABELS.get(pt, pt),
                   alpha=0.35, s=20, edgecolors="none")
    # 구종별 평균 마커
    for pt in available:
        p = df[df["PITCH_TYPE"] == pt].dropna(subset=["PFX_X", "PFX_Z"])
        mx, mz = p["PFX_X"].mean(), p["PFX_Z"].mean()
        ax.scatter(mx, mz, c=pitch_colors.get(pt, "#999"),
                   s=250, edgecolors="white", linewidths=2.5, zorder=5, marker="D")
        ax.annotate(DEFAULT_PITCH_LABELS.get(pt, pt),
                    (mx, mz), textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight="bold", color=pitch_colors.get(pt, "#333"))

    # SL-FS 사이 거리 표시
    if "FS" in available and "SL" in available:
        fs_mx = df[df["PITCH_TYPE"] == "FS"]["PFX_X"].mean()
        fs_mz = df[df["PITCH_TYPE"] == "FS"]["PFX_Z"].mean()
        sl_mx = df[df["PITCH_TYPE"] == "SL"]["PFX_X"].mean()
        sl_mz = df[df["PITCH_TYPE"] == "SL"]["PFX_Z"].mean()
        dist = np.sqrt((fs_mx - sl_mx) ** 2 + (fs_mz - sl_mz) ** 2)
        mid_x = (fs_mx + sl_mx) / 2
        mid_z = (fs_mz + sl_mz) / 2
        ax.annotate(f"SL↔FS 거리: {dist:.2f}ft",
                    xy=(mid_x, mid_z), fontsize=10, fontweight="bold",
                    color="#333", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F57F17"))
        ax.plot([fs_mx, sl_mx], [fs_mz, sl_mz],
                color="#F57F17", lw=1.5, ls="--", alpha=0.6)

    ax.axhline(0, color="#ccc", lw=0.8)
    ax.axvline(0, color="#ccc", lw=0.8)
    ax.set_xlabel("PFX_X (Horizontal Break, ft)", fontsize=11)
    ax.set_ylabel("PFX_Z (Induced Vertical Break, ft)", fontsize=11)
    ax.set_title("Movement Cluster — SL vs FS Separation", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)

    # ════════════════════════════════════════════
    # [1,1] 수직 낙차(PFX_Z) 분포 비교 — 겹침 확인
    # ════════════════════════════════════════════
    ax = axes[1, 1]
    for pt in available:
        p = df[df["PITCH_TYPE"] == pt]["PFX_Z"].dropna()
        if len(p) > 5:
            ax.hist(p, bins=30, alpha=0.5, color=pitch_colors.get(pt, "#999"),
                    label=f"{DEFAULT_PITCH_LABELS.get(pt, pt)} (μ={p.mean():.2f})",
                    edgecolor="white", density=True)
            # KDE 오버레이
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(p)
            x_range = np.linspace(p.min() - 0.2, p.max() + 0.2, 200)
            ax.plot(x_range, kde(x_range), color=pitch_colors.get(pt, "#999"),
                    lw=2.5, alpha=0.9)

    # SL과 FS의 PFX_Z 겹침 영역 표시
    if "FS" in available and "SL" in available:
        fs_z = df[df["PITCH_TYPE"] == "FS"]["PFX_Z"].dropna()
        sl_z = df[df["PITCH_TYPE"] == "SL"]["PFX_Z"].dropna()
        overlap_min = max(fs_z.min(), sl_z.min())
        overlap_max = min(fs_z.max(), sl_z.max())
        if overlap_min < overlap_max:
            ax.axvspan(overlap_min, overlap_max, alpha=0.1, color="#FFD600",
                       label=f"SL<->FS Duplicated Area")

    ax.set_xlabel("PFX_Z (Induced Vertical Break, ft)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Vertical Movement Distribution — SL vs FS, Duplicated Area Check", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # ════════════════════════════════════════════
    # [2,0] SL 로케이션 — 피안타 vs 아웃
    # ════════════════════════════════════════════
    ax = axes[2, 0]

    # 전체 SL 로케이션 (연하게)
    sl_all = sl_df.dropna(subset=["PLATE_X", "PLATE_Z"])
    ax.scatter(sl_all["PLATE_X"], sl_all["PLATE_Z"],
               c="#E0E0E0", s=10, alpha=0.3, label=f"All SL (n={len(sl_all)})")

    # 아웃 (초록)
    sl_out = sl_df[(sl_df["IS_AB"] == 1) & (sl_df["IS_HIT"] == 0)].dropna(subset=["PLATE_X", "PLATE_Z"])
    if len(sl_out) > 0:
        ax.scatter(sl_out["PLATE_X"], sl_out["PLATE_Z"],
                   c="#2E7D32", s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
                   label=f"Out (n={len(sl_out)})", marker="o")

    # 피안타 (빨강, 크게)
    sl_hit = sl_df[sl_df["IS_HIT"] == 1].dropna(subset=["PLATE_X", "PLATE_Z"])
    if len(sl_hit) > 0:
        ax.scatter(sl_hit["PLATE_X"], sl_hit["PLATE_Z"],
                   c="#C62828", s=120, alpha=0.9, edgecolors="white", linewidths=1.5,
                   label=f"Hit (n={len(sl_hit)})", marker="*", zorder=5)
        # 피안타 유형 텍스트
        for _, row in sl_hit.iterrows():
            event = str(row.get("PITCH_EVENTS", ""))
            short = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}.get(event, "")
            if short:
                ax.annotate(short, (row["PLATE_X"], row["PLATE_Z"]),
                            textcoords="offset points", xytext=(6, 6),
                            fontsize=8, fontweight="bold", color="#C62828")

    # 존 내/밖 피안타율 비교
    sl_in = sl_df[sl_df["IN_ZONE"] == 1]
    sl_out_zone = sl_df[sl_df["IN_ZONE"] == 0]
    in_ab = sl_in["IS_AB"].sum()
    in_h = sl_in["IS_HIT"].sum()
    in_avg = in_h / in_ab if in_ab > 0 else 0
    out_ab = sl_out_zone["IS_AB"].sum()
    out_h = sl_out_zone["IS_HIT"].sum()
    out_avg = out_h / out_ab if out_ab > 0 else 0

    info_text = (f"Zone Outter: .{int(in_avg*1000):03d} ({in_h}/{in_ab}AB)\n"
                 f"존 밖: .{int(out_avg*1000):03d} ({out_h}/{out_ab}AB)")
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
            va="top", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9))

    draw_strike_zone(ax, color="#333", lw=2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xlabel("Plate X (ft)")
    ax.set_ylabel("Plate Z (ft)")
    ax.set_title("Slider Location — by hit(★) vs out(●)", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)

    # ════════════════════════════════════════════
    # [2,1] SL 타구 품질 (Exit Velo vs Launch Angle)
    # ════════════════════════════════════════════
    ax = axes[2, 1]

    sl_inplay = sl_df[sl_df["IS_INPLAY"] == 1].dropna(subset=["LAUNCH_SPEED", "LAUNCH_ANGLE"])

    if len(sl_inplay) > 3:
        # SL 타구
        scatter_sl = ax.scatter(sl_inplay["LAUNCH_ANGLE"], sl_inplay["LAUNCH_SPEED"],
                                c=pitch_colors.get("SL", "#999"), s=50, alpha=0.7,
                                edgecolors="white", linewidths=0.5,
                                label=f"SL In-Play (n={len(sl_inplay)})")

        # 피안타만 강조
        sl_hit_ip = sl_inplay[sl_inplay["IS_HIT"] == 1]
        if len(sl_hit_ip) > 0:
            ax.scatter(sl_hit_ip["LAUNCH_ANGLE"], sl_hit_ip["LAUNCH_SPEED"],
                       c="#C62828", s=120, alpha=0.9, edgecolors="white", linewidths=1.5,
                       marker="*", label=f"SL Hits (n={len(sl_hit_ip)})", zorder=5)

        # 비교: FF, FS 인플레이 평균
        for cpt in ["FF", "FS"]:
            cp = df[(df["PITCH_TYPE"] == cpt) & (df["IS_INPLAY"] == 1)].dropna(
                subset=["LAUNCH_SPEED", "LAUNCH_ANGLE"])
            if len(cp) > 3:
                ax.scatter(cp["LAUNCH_ANGLE"].mean(), cp["LAUNCH_SPEED"].mean(),
                           c=pitch_colors.get(cpt, "#999"), s=200, marker="D",
                           edgecolors="white", linewidths=2, zorder=6,
                           label=f"{DEFAULT_PITCH_LABELS.get(cpt, cpt)} avg")

        # 배럴 존 표시 (EV≥98, LA 26~30)
        barrel_zone = mpatches.FancyBboxPatch((26, 98), 4, 14,
                                               boxstyle="round,pad=0.5",
                                               facecolor="#C62828", alpha=0.1,
                                               edgecolor="#C62828", lw=1.5)
        ax.add_patch(barrel_zone)
        ax.text(28, 112.5, "Barrel\nZone", ha="center", fontsize=9,
                color="#C62828", fontweight="bold", alpha=0.6)

        # SL 평균 EV, LA
        avg_ev = sl_inplay["LAUNCH_SPEED"].mean()
        avg_la = sl_inplay["LAUNCH_ANGLE"].mean()
        ax.axhline(avg_ev, color=pitch_colors.get("SL", "#999"), ls="--", lw=1, alpha=0.4)
        ax.axvline(avg_la, color=pitch_colors.get("SL", "#999"), ls="--", lw=1, alpha=0.4)
        ax.text(ax.get_xlim()[1] - 2, avg_ev + 0.5,
                f"SL avg EV: {avg_ev:.1f}", fontsize=9,
                color=pitch_colors.get("SL", "#999"), fontweight="bold", ha="right")

    else:
        ax.text(0.5, 0.5, "In-Play Data Lacked", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)

    ax.set_xlabel("Launch Angle (°)", fontsize=11)
    ax.set_ylabel("Exit Velocity (mph)", fontsize=11)
    ax.set_title("Pitching For Batting Quality — Exit Velo vs Launch Angle", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"[SliderDiag] Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════
# 콘솔 상세 리포트
# ══════════════════════════════════════════════════════════

def print_slider_report(df, stand_label):
    """슬라이더 관련 상세 수치를 콘솔에 출력합니다."""
    print(f"\n{'='*70}")
    print(f"  슬라이더 진단 리포트 — {stand_label}")
    print(f"{'='*70}")

    available = [pt for pt in ["FF", "FS", "SL"] if pt in df["PITCH_TYPE"].values]

    # 1. 구속 비교
    print(f"\n[1] 구속 비교")
    print(f"  {'구종':<10} {'avg':>7} {'max':>7} {'min':>7} {'std':>6}")
    print(f"  {'-'*40}")
    for pt in available:
        v = df[df["PITCH_TYPE"] == pt]["RELEASE_SPEED"].dropna()
        print(f"  {DEFAULT_PITCH_LABELS.get(pt, pt):<10} {v.mean():>6.1f}  {v.max():>6.1f}  {v.min():>6.1f}  {v.std():>5.1f}")

    if "FF" in available and "SL" in available:
        gap = df[df["PITCH_TYPE"] == "FF"]["RELEASE_SPEED"].mean() - \
              df[df["PITCH_TYPE"] == "SL"]["RELEASE_SPEED"].mean()
        print(f"  → FF-SL 구속차: {gap:.1f} mph")
    if "FS" in available and "SL" in available:
        gap2 = df[df["PITCH_TYPE"] == "FS"]["RELEASE_SPEED"].mean() - \
               df[df["PITCH_TYPE"] == "SL"]["RELEASE_SPEED"].mean()
        print(f"  → FS-SL 구속차: {gap2:.1f} mph")

    # 2. 무브먼트 비교
    print(f"\n[2] 무브먼트 비교")
    print(f"  {'구종':<10} {'PFX_X':>7} {'PFX_Z':>7} {'|PFX_X|':>7}")
    print(f"  {'-'*35}")
    for pt in available:
        p = df[df["PITCH_TYPE"] == pt]
        print(f"  {DEFAULT_PITCH_LABELS.get(pt, pt):<10} {p['PFX_X'].mean():>6.2f}  {p['PFX_Z'].mean():>6.2f}  {p['PFX_X'].abs().mean():>6.2f}")

    # 3. SL 존 안/밖 성적
    sl = df[df["PITCH_TYPE"] == "SL"]
    if len(sl) > 0:
        print(f"\n[3] 슬라이더 존 내/밖 성적")
        for zone_label, mask in [("존 안", sl["IN_ZONE"] == 1), ("존 밖", sl["IN_ZONE"] == 0)]:
            z = sl[mask]
            ab = z["IS_AB"].sum()
            h = z["IS_HIT"].sum()
            avg = h / ab if ab > 0 else 0
            sw = z["IS_SWING"].sum()
            whiff = z["IS_WHIFF"].sum() / sw * 100 if sw > 0 else 0
            print(f"  {zone_label}: {len(z)}투구, {ab}AB, {h}H, "
                  f"AVG .{int(avg*1000):03d}, Whiff {whiff:.1f}%")

    # 4. SL 피안타 상세
    sl_hits = sl[sl["IS_HIT"] == 1]
    if len(sl_hits) > 0:
        print(f"\n[4] 슬라이더 피안타 상세 ({len(sl_hits)}개)")
        for _, row in sl_hits.iterrows():
            ev = row.get("LAUNCH_SPEED", np.nan)
            la = row.get("LAUNCH_ANGLE", np.nan)
            px = row.get("PLATE_X", np.nan)
            pz = row.get("PLATE_Z", np.nan)
            event = row.get("PITCH_EVENTS", "?")
            in_z = "존안" if row.get("IN_ZONE", 0) == 1 else "존밖"
            print(f"  {event:<12} EV={ev:>5.1f} LA={la:>4.0f}° "
                  f"위치=({px:>+5.2f}, {pz:>4.2f}) [{in_z}]")

    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════
# 메인 실행
# ══════════════════════════════════════════════════════════

def slider_diagnosis(df: pd.DataFrame,
                     pitcher_name: str = "Pitcher",
                     pitch_colors: Optional[dict] = None,
                     save_dir: Optional[str] = ".",
                     dpi: int = 150):
    import os

    if pitch_colors is None:
        pitch_colors = DEFAULT_PITCH_COLORS.copy()

    df = prep_slider_analysis(df)

    splits = [
        ("L", "vs LHH (Left-Handed Hitters)", "_vsLHH"),
        ("R", "vs RHH (Right-Handed Hitters)", "_vsRHH"),
    ]

    for stand, label, suffix in splits:
        sub = df[df["STAND"] == stand].copy()
        n = len(sub)
        if n == 0:
            print(f"[SKIP] {label} — no data")
            continue

        full_label = f"{label}  ({n} pitches)"

        # 콘솔 리포트
        print_slider_report(sub, full_label)

        # 시각화
        sp = None
        if save_dir:
            sp = os.path.join(save_dir, f"slider_diagnosis{suffix}.png")

        plot_slider_diagnosis(sub, pitcher_name, full_label, pitch_colors,
                              save_path=sp, dpi=dpi)


# ══════════════════════════════════════════════════════════
# 실행 블록
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
        cursor = conn.cursor()

        sql = """
            SELECT *
            FROM TMP_TABLE
            WHERE PITCHER = 592332
        """
        cursor.execute(sql)
        raw_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
        print(f"Total rows fetched: {len(raw_df)}")

        clean_df = raw_df.dropna(
            subset=["PLATE_X", "PLATE_Z", "PITCH_TYPE",
                    "RELEASE_POS_X", "RELEASE_POS_Y", "RELEASE_POS_Z",
                    "VX0", "VY0", "VZ0",
                    "AX", "AY", "AZ",
                    "PFX_X", "PFX_Z", "RELEASE_SPIN_RATE"]
        )
        print(f"After cleaning: {len(clean_df)} rows")

        slider_diagnosis(
            clean_df,
            pitcher_name="Kevin Gausman",
            save_dir=".",
            dpi=150,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError occurred: {e}")
    finally:
        if "conn" in locals():
            conn.close()
            print("DB connection closed.")