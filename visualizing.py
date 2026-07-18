"""
투수 타석별 분석 — 투구 궤적(Side/Bird) + 투구 기록 테이블
"""

import oracledb as DB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SZ_L, SZ_R = -0.833, 0.833
SZ_B, SZ_T = 1.5, 3.5

FEATURES = [
    "PLATE_X", "PLATE_Z", "PFX_X", "PFX_Z",
    "RELEASE_SPEED", "RELEASE_SPIN_RATE", "SPIN_AXIS",
    "STAND", "PITCH_TYPE",
    "RELEASE_POS_X", "RELEASE_POS_Y", "RELEASE_POS_Z",
    "VX0", "VY0", "VZ0", "AX", "AY", "AZ",
]

PITCH_COLORS = {
    "FF": "#D83E23", "SI": "#7F77DD", "FC": "#d4a017",
    "SL": "#4E4D44", "SV": "#d03030", "ST": "#c02828",
    "CU": "#40b040", "CH": "#4090d0", "FS": "#1DD1D1",
}


def border_line():
    return plt.Rectangle(
        xy=[SZ_L, SZ_B], width=SZ_R - SZ_L, height=SZ_T - SZ_B,
        fill=False, linewidth=1.5
    )

def plot_pitch_trajectory(ab_df, axes_side, axes_bird):
    ab_df = ab_df.sort_values('PITCH_NUMBER')

    for _, row in ab_df.iterrows():
        x0, y0, z0 = row['RELEASE_POS_X'], row['RELEASE_POS_Y'], row['RELEASE_POS_Z']
        vx, vy, vz = row['VX0'], row['VY0'], row['VZ0']
        ax_val, ay, az = row['AX'], row['AY'], row['AZ']

        a_coef = 0.5 * ay
        b_coef = vy
        c_coef = y0 - 1.417
        disc = b_coef ** 2 - 4 * a_coef * c_coef
        if disc < 0:
            continue
        t_plate = (-b_coef - np.sqrt(disc)) / (2 * a_coef)

        t = np.linspace(0, t_plate, 100)
        traj_x = x0 + vx * t + 0.5 * ax_val * t ** 2
        traj_y = y0 + vy * t + 0.5 * ay * t ** 2
        traj_z = z0 + vz * t + 0.5 * az * t ** 2

        # ── PLATE_X, PLATE_Z 보정 ──
        # 물리 모델 끝점과 실제 값의 차이를 선형 보간으로 분배
        if pd.notna(row.get('PLATE_X')) and pd.notna(row.get('PLATE_Z')):
            offset_x = row['PLATE_X'] - traj_x[-1]
            offset_z = row['PLATE_Z'] - traj_z[-1]
            correction = np.linspace(0, 1, len(t))
            traj_x = traj_x + offset_x * correction
            traj_z = traj_z + offset_z * correction

        color = PITCH_COLORS.get(row['PITCH_TYPE'], '#999')
        label = f"#{int(row['PITCH_NUMBER'])} {row['PITCH_TYPE']}"

        axes_side.plot(traj_y, traj_z, color=color, linewidth=1.8, label=label)
        axes_side.scatter(traj_y[-1], traj_z[-1], color=color, s=60, zorder=5)

        axes_bird.plot(traj_x, traj_y, color=color, linewidth=1.8, label=label)
        axes_bird.scatter(traj_x[-1], traj_y[-1], color=color, s=60, zorder=5)

    axes_side.plot([1.417, 1.417], [SZ_B, SZ_T], color='black', linewidth=2)
    axes_side.set_xlabel('Y — Distance to Plate (ft)')
    axes_side.set_ylabel('Z — Height (ft)')
    axes_side.set_title('Side View')
    axes_side.legend(fontsize=8, loc='upper right')
    axes_side.invert_xaxis()
    axes_side.set_xlim(0, 60)
    axes_side.set_ylim(0, 7)
    axes_side.set_aspect(3.5)

    axes_bird.plot([-0.83, 0.83], [1.417, 1.417], color='black', linewidth=2)
    axes_bird.set_xlabel('Y — Distance to Plate (ft)')
    axes_bird.set_ylabel('X — Horizontal (ft)')
    axes_bird.set_title("Bird's Eye View")
    axes_bird.legend(fontsize=8, loc='upper right')
    axes_bird.set_xlim(-3, 3)
    axes_bird.set_ylim(0, 60)
    axes_bird.set_aspect(.33)

def plot_pitch_analysis(df, pitcher_name="Pitcher"):
    for (game_pk, ab_num), ab_df in df.groupby(['GAME_PK', 'AT_BAT_NUMBER']):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                                 gridspec_kw={'width_ratios': [1, 1]})
        # 좌측 2패널: 궤적
        plot_pitch_trajectory(ab_df, axes[0], axes[1])

        fig.suptitle(f"{pitcher_name} | Game: {game_pk} | AB: {ab_num}",
                     fontsize=16, fontweight='bold')
        fig.tight_layout()
        plt.show()

def calc_vaa(row):
    """홈플레이트 도달 시점의 수직 접근 각도(VAA)를 계산합니다."""
    x0, y0, z0 = row['RELEASE_POS_X'], row['RELEASE_POS_Y'], row['RELEASE_POS_Z']
    vx, vy, vz = row['VX0'], row['VY0'], row['VZ0']
    ax, ay, az = row['AX'], row['AY'], row['AZ']

    # 홈플레이트(y=1.417ft) 도달 시간
    a_coef = 0.5 * ay
    b_coef = vy
    c_coef = y0 - 1.417
    disc = b_coef ** 2 - 4 * a_coef * c_coef
    if disc < 0:
        return np.nan
    t = (-b_coef - np.sqrt(disc)) / (2 * a_coef)

    # 도달 시점의 속도 벡터
    vz_final = vz + az * t
    vy_final = vy + ay * t

    # VAA = arctan(수직속도 / 수평속도)
    vaa = np.degrees(np.arctan2(vz_final, -vy_final))
    return round(vaa, 2)

def FF_analysis(df: pd.DataFrame):
    FF_DF = df[df['PITCH_TYPE'] == 'FF'].copy()
    FF_DF['VAA'] = FF_DF.apply(calc_vaa, axis=1)

    FF_DF['speed_group'] = pd.cut(
        FF_DF['RELEASE_SPEED'],
        bins=[0, 90, 93, 95, 97, 110],
        labels=['~90', '90~93', '93~95', '95~97', '97~'],
        right=True
    )

    # ====================================================
    # 1. 구속대별 투구 유효율
    # ====================================================
    swing_desc = ['swinging_strike', 'swinging_strike_blocked',
                  'foul', 'foul_tip', 'hit_into_play',
                  'hit_into_play_no_out', 'hit_into_play_score']
    whiff_desc = ['swinging_strike', 'swinging_strike_blocked']
    f_hit = ['single', 'double', 'triple', 'home_run']
    non_ab_events = ['walk', 'hit_by_pitch', 'sac_bunt', 'sac_fly', 'intent_walk']

    print("=" * 80)
    print("구속대별 투구 유효율")
    print("=" * 80)
    for name, d_df in FF_DF.groupby('speed_group'):
        non_final = d_df[d_df['PITCH_EVENTS'].isna()]
        n_total = len(d_df)

        whiff  = non_final['PITCH_DESCRIPTION'].isin(whiff_desc).mean()
        foul   = non_final['PITCH_DESCRIPTION'].isin(['foul', 'foul_tip']).mean()
        called = non_final['PITCH_DESCRIPTION'].isin(['called_strike']).mean()
        swing_rate = non_final['PITCH_DESCRIPTION'].isin(swing_desc).mean()

        print(f"{name:>6} (n={n_total:>6,}) | "
              f"스윙률: {swing_rate:.3f} | 헛스윙: {whiff:.3f} | "
              f"파울: {foul:.3f} | 루킹: {called:.3f}")

    # ====================================================
    # 2. PFX_Z / VAA 분포 시각화
    # ====================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.histplot(data=FF_DF, x='PFX_Z', bins=50, ax=axes[0], color='#3266ad')
    axes[0].axvline(FF_DF['PFX_Z'].median(), color='red', ls='--',
                    label=f"median: {FF_DF['PFX_Z'].median():.1f}")
    axes[0].set_title('Induced Vertical Break (PFX_Z)')
    axes[0].set_xlabel('PFX_Z (inches)')
    axes[0].legend()

    sns.histplot(data=FF_DF, x='VAA', bins=50, ax=axes[1], color='#D85A30')
    axes[1].axvline(FF_DF['VAA'].median(), color='red', ls='--',
                    label=f"median: {FF_DF['VAA'].median():.1f}°")
    axes[1].set_title('Vertical Approach Angle (VAA)')
    axes[1].set_xlabel('VAA (degrees)')
    axes[1].legend()

    sns.scatterplot(data=FF_DF, x='PFX_Z', y='VAA', alpha=0.3, s=10,
                    ax=axes[2], color='#2E7D32')
    axes[2].set_title('PFX_Z vs VAA')
    axes[2].set_xlabel('PFX_Z (inches)')
    axes[2].set_ylabel('VAA (degrees)')

    fig.suptitle('Four-Seam Fastball: Rising Effect Metrics',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()

    # ====================================================
    # 3. 4분면 분석 — VAA × PFX_Z 조합별 성적
    # ====================================================
    vaa_med = FF_DF['VAA'].median()
    pfx_med = FF_DF['PFX_Z'].median()

    conditions = [
        (FF_DF['VAA'] > vaa_med) & (FF_DF['PFX_Z'] > pfx_med),
        (FF_DF['VAA'] > vaa_med) & (FF_DF['PFX_Z'] <= pfx_med),
        (FF_DF['VAA'] <= vaa_med) & (FF_DF['PFX_Z'] > pfx_med),
        (FF_DF['VAA'] <= vaa_med) & (FF_DF['PFX_Z'] <= pfx_med),
    ]
    labels = ['A: Flat+High IVB', 'B: Flat+Low IVB',
              'C: Steep+High IVB', 'D: Steep+Low IVB']
    FF_DF['quad'] = np.select(conditions, labels, default='Unknown')

    print("\n" + "=" * 80)
    print(f"4분면 분석 (기준 — VAA: {vaa_med:.2f}° / PFX_Z: {pfx_med:.2f}in)")
    print("=" * 80)

    quad_results = []
    for name, g in FF_DF.groupby('quad'):
        non_final = g[g['PITCH_EVENTS'].isna()]
        swing_n = non_final['PITCH_DESCRIPTION'].isin(swing_desc).sum()
        whiff_n = non_final['PITCH_DESCRIPTION'].isin(whiff_desc).sum()
        whiff_rate = whiff_n / swing_n if swing_n > 0 else 0
        swing_rate = non_final['PITCH_DESCRIPTION'].isin(swing_desc).mean()
        called_rate = non_final['PITCH_DESCRIPTION'].isin(['called_strike']).mean()

        final = g.dropna(subset=['PITCH_EVENTS'])
        ab = final[~final['PITCH_EVENTS'].isin(non_ab_events)]
        baa = ab['PITCH_EVENTS'].isin(f_hit).mean() if len(ab) > 0 else 0

        quad_results.append({
            'quad': name, 'n': len(g),
            'swing_rate': swing_rate,
            'whiff_rate': whiff_rate,
            'called_rate': called_rate,
            'baa': baa
        })
        print(f"{name:>22} (n={len(g):>6,}) | "
              f"스윙률: {swing_rate:.3f} | Whiff%: {whiff_rate:.3f} | "
              f"루킹: {called_rate:.3f} | BAA: {baa:.3f}")

    quad_df = pd.DataFrame(quad_results)

    
    # ====================================

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # VZ0을 결정하는 코치 조절 가능 변수는?
    controllable = ['RELEASE_POS_Z', 'RELEASE_SPEED', 'PFX_Z']
    controllable = [c for c in controllable if c in FF_DF.columns]

    temp = FF_DF[controllable + ['VZ0']].dropna()
    X = temp[controllable]
    y = temp['VZ0']

    print("=" * 60)
    print("VZ0을 결정하는 조절 가능 변수")
    print("=" * 60)

    print("\n[개별 R²]")
    for col in controllable:
        model = LinearRegression().fit(X[[col]], y)
        print(f"  {col:>22}: {model.score(X[[col]], y):.3f}")

    model = LinearRegression().fit(X, y)
    print(f"\n[결합 R²]: {model.score(X, y):.3f}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_s = LinearRegression().fit(X_scaled, y)
    print(f"\n[표준화 회귀계수 — 기여도]")
    for col, coef in zip(controllable, model_s.coef_):
        print(f"  {col:>22}: {coef:>+.3f}")

    print(FF_DF[controllable].corr().round(3))

    # ===========================================================


    # ====================================================
    # 4. 4분면 시각화 — 산점도 + 성적 바차트
    # ====================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    quad_colors = {
        'A: Flat+High IVB': '#D83E23',
        'B: Flat+Low IVB':  '#F4A261',
        'C: Steep+High IVB':'#3266AD',
        'D: Steep+Low IVB': '#89CFF0',
    }

    # 4-1) VAA vs PFX_Z 산점도에 4분면 표시
    for quad_name, color in quad_colors.items():
        sub = FF_DF[FF_DF['quad'] == quad_name]
        axes[0].scatter(sub['PFX_Z'], sub['VAA'], s=8, alpha=0.25,
                        color=color, label=quad_name)
    axes[0].axhline(vaa_med, color='black', ls='--', lw=1, alpha=0.5)
    axes[0].axvline(pfx_med, color='black', ls='--', lw=1, alpha=0.5)
    axes[0].set_xlabel('PFX_Z (inches)')
    axes[0].set_ylabel('VAA (degrees)')
    axes[0].set_title('Quadrant Map')
    axes[0].legend(fontsize=8, markerscale=3)

    # 4-2) Whiff Rate 비교
    bars = axes[1].bar(quad_df['quad'], quad_df['whiff_rate'],
                       color=[quad_colors[q] for q in quad_df['quad']],
                       edgecolor='white')
    for bar, val in zip(bars, quad_df['whiff_rate']):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'{val:.1%}', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Whiff Rate')
    axes[1].set_title('Whiff Rate by Quadrant')
    axes[1].tick_params(axis='x', rotation=20)

    # 4-3) BAA 비교
    bars = axes[2].bar(quad_df['quad'], quad_df['baa'],
                       color=[quad_colors[q] for q in quad_df['quad']],
                       edgecolor='white')
    for bar, val in zip(bars, quad_df['baa']):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f'.{int(val * 1000):03d}', ha='center', fontsize=10, fontweight='bold')
    axes[2].axhline(0.250, color='#999', ls='--', lw=0.8)
    axes[2].set_ylabel('BAA')
    axes[2].set_title('Batting Avg Against by Quadrant')
    axes[2].tick_params(axis='x', rotation=20)

    fig.suptitle('Four-Seam Fastball: VAA × PFX_Z Quadrant Analysis',
                 fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()

    FF_DF['rel_height_group'] = pd.qcut(
        FF_DF['RELEASE_POS_Z'], q=3,
        labels=['Low Slot', 'Mid Slot', 'High Slot']
    )

    FF_DF['height_zone'] = pd.cut(
        FF_DF['PLATE_Z'],
        bins=[0, SZ_B, (SZ_B + SZ_T) / 2, SZ_T, 6],
        labels=['Below', 'Low Zone', 'High Zone', 'Above']
    )

    print("릴리스 높이 × 존 높이별 평균 VAA")
    print("=" * 65)
    for slot in ['Low Slot', 'Mid Slot', 'High Slot']:
        print(f"\n  [{slot}]")
        sub = FF_DF[FF_DF['rel_height_group'] == slot]
        for zone in ['Above', 'High Zone', 'Low Zone', 'Below']:
            z = sub[sub['height_zone'] == zone]['VAA']
            if len(z) > 10:
                print(f"    {zone:>10}: VAA {z.mean():>6.2f}° (n={len(z):,})")
    FF_DF = FF_DF[FF_DF['P_THROWS'] == 'R'].copy()
    print(FF_DF[['AX', 'SPIN_AXIS']].corr())            

def simple_statistics(df : pd.DataFrame, pitcher_name : str):
    # ------------------ 전체 ------------------------------
    all_df = df[df['PITCH_TYPE'].isin(['FF', 'CH', 'SL'])]
    all_whiff_df = all_df[all_df['PITCH_DESCRIPTION'].isin([
        'foul', 'hit_into_play', 'swinging_strike', 'swinging_strike_blocked', 
        'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
    ])].copy()
    
    all_whiff_df['WHIFF'] = np.where(
        all_whiff_df['PITCH_DESCRIPTION'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip', 'bunt_foul_tip']), 1, 0
    )

    all_whiff_df = all_whiff_df.groupby(['PITCH_TYPE'])['WHIFF'].agg(
        WHIFF_RATE='mean', 
        SWING_COUNT='count'
    )

    all_out_df = all_df.dropna(subset=['PITCH_EVENTS']).copy()
    all_out_df['OUT_RATE'] = np.where(
        all_out_df['PITCH_EVENTS'].isin(['field_out', 'strikeout', 'ground_into_double_play', 'force_out',
        'double_play', 'fielders_choice_out', 'strikeout_double_play', 'triple_play']), 1, 0
    )
    all_out_df = all_out_df.groupby(['PITCH_TYPE'])['OUT_RATE'].agg(
        OUT_RATE = 'mean',
        OUT_COUNT = 'count'
    )
    all_whiff_df = pd.merge(
        left=all_whiff_df, right=all_out_df, on= ['PITCH_TYPE']
    )
    
    print(f"_____ 리그 평균 ____________ \n {all_whiff_df} \n")

    # ----------- 메이슨 밀러 -------------------------
    MM_whiff_df = df[df['PITCH_DESCRIPTION'].isin([
        'foul', 'hit_into_play', 'swinging_strike', 'swinging_strike_blocked', 
        'foul_tip', 'foul_bunt', 'missed_bunt', 'bunt_foul_tip'
    ])].copy()
    MM_whiff_df = MM_whiff_df[MM_whiff_df['PITCHER'] == 695243]
    MM_whiff_df['WHIFF'] = np.where(
        MM_whiff_df['PITCH_DESCRIPTION'].isin(['swinging_strike', 'swinging_strike_blocked', 'foul_tip', 'bunt_foul_tip']), 1, 0
    )

    MM_whiff_df = MM_whiff_df.groupby(['PITCH_TYPE'])['WHIFF'].agg(
        WHIFF_RATE='mean', 
        SWING_COUNT='count'
    )

    MM_out_df = all_df.dropna(subset=['PITCH_EVENTS']).copy()
    MM_out_df = MM_out_df[MM_out_df['PITCHER'] == 695243]
    MM_out_df['OUT_RATE'] = np.where(
        MM_out_df['PITCH_EVENTS'].isin(['field_out', 'strikeout', 'ground_into_double_play', 'force_out',
        'double_play', 'fielders_choice_out', 'strikeout_double_play', 'triple_play']), 1, 0
    )
    MM_out_df = MM_out_df.groupby(['PITCH_TYPE'])['OUT_RATE'].agg(
        OUT_RATE = 'mean',
        OUT_COUNT = 'count'
    )
    MM_whiff_df = pd.merge(
        left=MM_whiff_df, right=MM_out_df, on= ['PITCH_TYPE']
    )

    print(f"_____  메이슨 밀러 ______ \n{MM_whiff_df}\n")

    
    # plot = sns.kdeplot(
    #     data=FF_DF, x= 'RELEASE_SPEED', ax= axes
    # )

    # axes.legend()
    # plt.show()

from matplotlib.widgets import Slider, RadioButtons

def interactive_trajectory(df, pitcher_name="Pitcher"):
    """구종 선택 + 투구 번호 슬라이더로 개별 궤적을 탐색합니다."""
    df = df.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER']).copy()

    # ── 구종별 투구 목록 생성 (오름차순 넘버링) ──
    pitch_types = sorted(df['PITCH_TYPE'].unique().tolist())
    pitch_dict = {}
    for pt in pitch_types:
        sub = df[df['PITCH_TYPE'] == pt].reset_index(drop=True)
        sub['SEQ'] = range(1, len(sub) + 1)
        pitch_dict[pt] = sub

    current_pt = [pitch_types[0]]

    # ── 그림 + 여백 확보 ──
    fig = plt.figure(figsize=(18, 9))
    fig.subplots_adjust(left=0.08, right=0.75, bottom=0.18, top=0.90)

    ax_side = fig.add_axes([0.08, 0.22, 0.30, 0.65])
    ax_bird = fig.add_axes([0.42, 0.22, 0.30, 0.65])
    ax_radio = fig.add_axes([0.78, 0.45, 0.12, 0.30])
    ax_slider = fig.add_axes([0.15, 0.06, 0.50, 0.04])

    radio = RadioButtons(ax_radio, pitch_types, active=0)
    for label in radio.labels:
        label.set_fontsize(11)
    ax_radio.set_title('Pitch Type', fontsize=12, fontweight='bold')

    slider = Slider(ax_slider, 'Pitch #', 1, len(pitch_dict[current_pt[0]]),
                    valinit=1, valstep=1)

    # ── 단일 투구 궤적 그리기 ──
    def draw(row):
        ax_side.cla()
        ax_bird.cla()

        x0, y0, z0 = row['RELEASE_POS_X'], row['RELEASE_POS_Y'], row['RELEASE_POS_Z']
        vx, vy, vz = row['VX0'], row['VY0'], row['VZ0']
        ax_val, ay, az = row['AX'], row['AY'], row['AZ']

        a_coef = 0.5 * ay
        b_coef = vy
        c_coef = y0 - 1.417
        disc = b_coef ** 2 - 4 * a_coef * c_coef
        if disc < 0:
            return
        t_plate = (-b_coef - np.sqrt(disc)) / (2 * a_coef)

        t = np.linspace(0, t_plate, 100)
        traj_x = x0 + vx * t + 0.5 * ax_val * t ** 2
        traj_y = y0 + vy * t + 0.5 * ay * t ** 2
        traj_z = z0 + vz * t + 0.5 * az * t ** 2

        if pd.notna(row.get('PLATE_X')) and pd.notna(row.get('PLATE_Z')):
            correction = np.linspace(0, 1, len(t))
            traj_x = traj_x + (row['PLATE_X'] - traj_x[-1]) * correction
            traj_z = traj_z + (row['PLATE_Z'] - traj_z[-1]) * correction

        color = PITCH_COLORS.get(row['PITCH_TYPE'], '#999')

        # Side View
        ax_side.plot(traj_y, traj_z, color=color, linewidth=2)
        ax_side.scatter(traj_y[-1], traj_z[-1], color=color, s=80, zorder=5)
        ax_side.plot([1.417, 1.417], [SZ_B, SZ_T], color='black', linewidth=2)
        ax_side.set_xlim(60, 0)
        ax_side.set_ylim(0, 7)
        ax_side.set_aspect(3.5)
        ax_side.set_xlabel('Y — Distance to Plate (ft)')
        ax_side.set_ylabel('Z — Height (ft)')
        ax_side.set_title('Side View')

        # Bird View
        ax_bird.plot(traj_x, traj_y, color=color, linewidth=2)
        ax_bird.scatter(traj_x[-1], traj_y[-1], color=color, s=80, zorder=5)
        ax_bird.plot([-0.83, 0.83], [1.417, 1.417], color='black', linewidth=2)
        ax_bird.set_xlim(-3, 3)
        ax_bird.set_ylim(60, 0)
        ax_bird.invert_xaxis()
        ax_bird.set_aspect(0.33)
        ax_bird.set_xlabel('X — Horizontal (ft)')
        ax_bird.set_ylabel('Y — Distance to Plate (ft)')
        ax_bird.set_title("Bird's Eye View")

        # 투구 정보 텍스트
        info = (f"{row['PITCH_TYPE']} | {row['RELEASE_SPEED']:.1f}mph | "
                f"PFX_X: {row['PFX_X']:.1f} | PFX_Z: {row['PFX_Z']:.1f}\n"
                f"Game: {int(row['GAME_PK'])} | AB: {int(row['AT_BAT_NUMBER'])} | "
                f"Pitch: {int(row['PITCH_NUMBER'])}")
        fig.suptitle(f"{pitcher_name}\n{info}", fontsize=13, fontweight='bold')

        fig.canvas.draw_idle()

    # ── 슬라이더 업데이트 ──
    def on_slider(val):
        idx = int(val) - 1
        sub = pitch_dict[current_pt[0]]
        if 0 <= idx < len(sub):
            draw(sub.iloc[idx])

    # ── 라디오 버튼 업데이트 ──
    def on_radio(label):
        current_pt[0] = label
        sub = pitch_dict[label]
        slider.valmax = len(sub)
        slider.ax.set_xlim(1, len(sub))
        slider.set_val(1)
        draw(sub.iloc[0])

    slider.on_changed(on_slider)
    radio.on_clicked(on_radio)

    draw(pitch_dict[current_pt[0]].iloc[0])
    plt.show()

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="1234", dsn="localhost:1521/xe")
        cursor = conn.cursor()

        sql = """
            SELECT *
            FROM TMP_TABLE
            WHERE PITCHER = 695243
        """

        cursor.execute(sql)
        raw = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
        df = raw.dropna(subset=FEATURES).copy()
        df = df.sort_values(
            by=["GAME_PK", "AT_BAT_NUMBER", "PITCH_NUMBER"],
            ascending=[True, True, True]
        )
        print(f"{len(df)} rows")

        
        if len(df) > 0:
        #    FF_analysis(df)
        #   plot_pitch_analysis(df, pitcher_name="Mason Miller")
            interactive_trajectory(df, pitcher_name="Mason Miller")
        else:
            print("NO DATA FOUND")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()