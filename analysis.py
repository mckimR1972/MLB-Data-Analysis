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
    "VX0", "VY0", "VZ0", "AX", "AY", "AZ", "ARM_ANGLE"
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

    FF_DF['vaa_group'] = pd.cut(
        FF_DF['VAA'],
        bins=[-12, -7, -6, -5, -4, -3, 0],
        labels=['~-7°', '-7~-6°', '-6~-5°', '-5~-4°', '-4~-3°', '-3°~']
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # 1) VAA 구간별 PLATE_Z 박스플롯
    sns.boxplot(
        data=FF_DF, x='vaa_group', y='PLATE_Z', ax=axes[0],
        palette='RdYlBu'
    )
    axes[0].axhline(SZ_T, color='red', ls='--', lw=1, alpha=0.5, label='Zone Top')
    axes[0].axhline(SZ_B, color='blue', ls='--', lw=1, alpha=0.5, label='Zone Bot')
    axes[0].set_title('PLATE_Z Distribution by VAA Group')
    axes[0].set_xlabel('VAA Group')
    axes[0].set_ylabel('PLATE_Z (ft)')
    axes[0].legend(fontsize=8)

    # 2) VAA 구간별 PLATE_Z 히스토그램 (겹쳐서)
    vaa_groups = ['~-7°', '-6~-5°', '-4~-3°']
    colors = ['#3266AD', '#888888', '#D83E23']
    for grp, color in zip(vaa_groups, colors):
        sub = FF_DF[FF_DF['vaa_group'] == grp]
        axes[1].hist(sub['PLATE_Z'], bins=40, alpha=0.5,
                    color=color, label=f'{grp} (n={len(sub):,})', density=True)
    axes[1].axvline(SZ_T, color='red', ls='--', lw=1, alpha=0.5)
    axes[1].axvline(SZ_B, color='blue', ls='--', lw=1, alpha=0.5)
    axes[1].set_title('PLATE_Z Density: Steep vs Mid vs Flat')
    axes[1].set_xlabel('PLATE_Z (ft)')
    axes[1].set_ylabel('Density')
    axes[1].legend(fontsize=9)

    # 3) VAA 구간별 평균 PLATE_Z + 투구 수
    summary = FF_DF.groupby('vaa_group').agg(
        mean_z=('PLATE_Z', 'mean'),
        count=('PLATE_Z', 'count')
    ).reset_index()

    bars = axes[2].bar(summary['vaa_group'], summary['mean_z'],
                    color=plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(summary))),
                    edgecolor='white')
    for bar, (_, row) in zip(bars, summary.iterrows()):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f'{row["mean_z"]:.2f}ft\n(n={int(row["count"]):,})',
                    ha='center', fontsize=9, fontweight='bold')
    axes[2].axhline(SZ_T, color='red', ls='--', lw=1, alpha=0.5)
    axes[2].axhline(SZ_B, color='blue', ls='--', lw=1, alpha=0.5)
    axes[2].set_title('Mean PLATE_Z by VAA Group')
    axes[2].set_xlabel('VAA Group')
    axes[2].set_ylabel('Mean PLATE_Z (ft)')

    fig.suptitle('Four-Seam Fastball: Where Do Pitches End Up by VAA?',
                fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.show()    
          

def FF_Spin_Axis(df: pd.DataFrame):
    FF_DF = df[df['PITCH_TYPE'] == 'FF'].copy()
    l = FF_DF[FF_DF['P_THROWS'] == 'L'].copy()
    r = FF_DF[FF_DF['P_THROWS'] == 'R'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].set_xlim(50, 300)
    ax_plot = sns.scatterplot(
        data= r, x='SPIN_AXIS', y= 'AX', s=20, ax=axes[0], c= 'Red'
    )

    vx0_plot = sns.scatterplot(
        data= r, x='SPIN_AXIS', y= 'ARM_ANGLE', s=20, ax=axes[1], c= 'Navy'
    ) 

    print(r[['SPIN_AXIS', 'ARM_ANGLE', 'AX']].corr())
    # 우투수 기준, 회전각이 180도에서 커질수록 우타 몸쪽으로 가속도 붙음 

    plt.show()

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

def plot_vaa_platez_whiff(df : pd.DataFrame):
    FF_DF = df[df['PITCH_TYPE'] == 'FF'].copy()
    FF_DF['VAA'] = FF_DF.apply(calc_vaa, axis=1)
    FF_DF = FF_DF.dropna(subset=['VAA'])

    swing_desc = ['swinging_strike', 'swinging_strike_blocked',
                  'foul', 'foul_tip', 'hit_into_play']
    whiff_desc = ['swinging_strike', 'swinging_strike_blocked']

    non_final = FF_DF[FF_DF['PITCH_EVENTS'].isna()].copy()
    non_final['IS_SWING'] = non_final['PITCH_DESCRIPTION'].isin(swing_desc).astype(int)
    non_final['IS_WHIFF'] = non_final['PITCH_DESCRIPTION'].isin(whiff_desc).astype(int)


    # ── 구간 분할 ──
    non_final['vaa_group'] = pd.cut(
        non_final['VAA'],
        bins=[-12, -7, -6, -5, -4, -3, 0],
        labels=['~-7°', '-7~-6°', '-6~-5°', '-5~-4°', '-4~-3°', '-3°~']
    )
    non_final['z_group'] = pd.cut(
        non_final['PLATE_Z'],
        bins=[0, SZ_B, (SZ_B + SZ_T) / 2, SZ_T, 6],
        labels=['Below', 'Low Zone', 'High Zone', 'Above']
    )
    

    # ── Whiff Rate 계산 ──
    grouped = non_final.groupby(['z_group', 'vaa_group']).agg(
        swings=('IS_SWING', 'sum'),
        whiffs=('IS_WHIFF', 'sum'),
        n=('IS_SWING', 'count')
    ).reset_index()
    grouped['whiff_rate'] = np.where(
        grouped['swings'] >= 10,
        grouped['whiffs'] / grouped['swings'],
        np.nan
    )

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # ── 1) 히트맵 ──
    pivot = grouped.pivot(index='z_group', columns='vaa_group', values='whiff_rate')
    pivot = pivot.reindex(['Above', 'High Zone', 'Low Zone', 'Below'])

    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlBu_r', ax=axes[0],
                vmin=0.05, vmax=0.35, linewidths=1, linecolor='white',
                cbar_kws={'label': 'Whiff Rate'})
    axes[0].set_title('Whiff Rate: PLATE_Z × VAA', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Pitch Height')
    axes[0].set_xlabel('VAA Group')

    # ── 2) 라인차트 — 숫자 x좌표로 강제 정렬 ──
    vaa_order = ['~-7°', '-7~-6°', '-6~-5°', '-5~-4°', '-4~-3°', '-3°~']
    zones = ['Above', 'High Zone', 'Low Zone', 'Below']
    zone_colors = {'Above': '#D83E23', 'High Zone': '#F4A261',
                'Low Zone': '#3266AD', 'Below': '#89CFF0'}

    for zone in zones:
        sub = grouped[grouped['z_group'] == zone].copy()
        sub['vaa_sort'] = sub['vaa_group'].map({v: i for i, v in enumerate(vaa_order)})
        sub = sub.sort_values('vaa_sort').dropna(subset=['whiff_rate'])

        axes[1].plot(sub['vaa_sort'], sub['whiff_rate'],
                    marker='o', linewidth=2, markersize=8,
                    color=zone_colors[zone], label=zone)
        for _, row in sub.iterrows():
            axes[1].annotate(f"{row['whiff_rate']:.1%}",
                            (row['vaa_sort'], row['whiff_rate']),
                            textcoords='offset points', xytext=(0, 10),
                            ha='center', fontsize=8)

    axes[1].set_xticks(range(len(vaa_order)))
    axes[1].set_xticklabels(vaa_order)
    axes[1].set_title('Whiff Rate Trend by Zone Height', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('VAA Group (Steep → Flat)')
    axes[1].set_ylabel('Whiff Rate')
    axes[1].legend(fontsize=9)

    # ── 3) 우측 — High Zone과 Low Zone 직접 비교 (그룹 바차트) ──
    high_data = grouped[grouped['z_group'] == 'High Zone'].copy()
    high_data['vaa_sort'] = high_data['vaa_group'].map({v: i for i, v in enumerate(vaa_order)})
    high_data = high_data.sort_values('vaa_sort').dropna(subset=['whiff_rate'])

    low_data = grouped[grouped['z_group'] == 'Low Zone'].copy()
    low_data['vaa_sort'] = low_data['vaa_group'].map({v: i for i, v in enumerate(vaa_order)})
    low_data = low_data.sort_values('vaa_sort').dropna(subset=['whiff_rate'])

    common_vaa = sorted(set(high_data['vaa_sort']) & set(low_data['vaa_sort']))
    w = 0.35
    x = np.arange(len(common_vaa))

    h_vals = [high_data[high_data['vaa_sort'] == v]['whiff_rate'].values[0] for v in common_vaa]
    l_vals = [low_data[low_data['vaa_sort'] == v]['whiff_rate'].values[0] for v in common_vaa]
    labels = [vaa_order[v] for v in common_vaa]

    bars_h = axes[2].bar(x - w/2, h_vals, w, color='#F4A261', label='High Zone', edgecolor='white')
    bars_l = axes[2].bar(x + w/2, l_vals, w, color='#3266AD', label='Low Zone', edgecolor='white')

    for bar, val in zip(bars_h, h_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', fontsize=9, fontweight='bold', color='#F4A261')
    for bar, val in zip(bars_l, l_vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', fontsize=9, fontweight='bold', color='#3266AD')

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title('High Zone vs Low Zone Whiff Rate', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('VAA Group (Steep → Flat)')
    axes[2].set_ylabel('Whiff Rate')
    axes[2].legend(fontsize=9)

    # ----------------------------------------------------
    # ── OPS 계산 (final 기반, 별도 집계) ──
    non_ab_events = ['walk', 'hit_by_pitch', 'sac_bunt', 'sac_fly',
                     'intent_walk', 'catcher_interf']
    hit_events = ['single', 'double', 'triple', 'home_run']
    final = FF_DF[FF_DF['PITCH_EVENTS'].notna()].copy()
    
    final['vaa_group'] = pd.cut(
        final['VAA'],
        bins=[-12, -7, -6, -5, -4, -3, 0],
        labels=['~-7°', '-7~-6°', '-6~-5°', '-5~-4°', '-4~-3°', '-3°~']
    )
    final['z_group'] = pd.cut(
        final['PLATE_Z'],
        bins=[0, SZ_B, (SZ_B + SZ_T) / 2, SZ_T, 6],
        labels=['Below', 'Low Zone', 'High Zone', 'Above']
    )

    final['IS_HIT'] = final['PITCH_EVENTS'].isin(hit_events).astype(int)
    final['IS_AB'] = (~final['PITCH_EVENTS'].isin(non_ab_events)).astype(int)
    final['IS_ON_BASE'] = final['PITCH_EVENTS'].isin(
        hit_events + ['walk', 'hit_by_pitch', 'intent_walk']).astype(int)
    final['IS_PA'] = 1

    conditions = [
        final['PITCH_EVENTS'] == 'single',
        final['PITCH_EVENTS'] == 'double',
        final['PITCH_EVENTS'] == 'triple',
        final['PITCH_EVENTS'] == 'home_run',
    ]
    final['TOTAL_BASES'] = np.select(conditions, [1, 2, 3, 4], default=0)

    final_grouped = final.groupby(['z_group', 'vaa_group']).agg(
        hits=('IS_HIT', 'sum'),
        ab=('IS_AB', 'sum'),
        ob=('IS_ON_BASE', 'sum'),
        pa=('IS_PA', 'sum'),
        tb=('TOTAL_BASES', 'sum'),
    ).reset_index()

    final_grouped['SLG'] = np.where(final_grouped['ab'] >= 10,
                                     final_grouped['tb'] / final_grouped['ab'], np.nan)
    # final_grouped['OBP'] = np.where(final_grouped['pa'] >= 10,
    #                                  final_grouped['ob'] / final_grouped['pa'], np.nan)
    # final_grouped['OPS'] = final_grouped['OBP'] + final_grouped['SLG']

    # ── 별도 창: OPS 히트맵 ──
    new_fig, new_axes = plt.subplots(1, 2, figsize=(18, 7))

    # pivot_ops = final_grouped.pivot(index='z_group', columns='vaa_group', values='OPS')
    # pivot_ops = pivot_ops.reindex(['Above', 'High Zone', 'Low Zone', 'Below'])

    # sns.heatmap(pivot_ops, annot=True, fmt='.3f', cmap='RdYlBu', ax=new_axes[0],
    #             vmin=0.3, vmax=1.0, linewidths=1, linecolor='white',
    #             cbar_kws={'label': 'OPS'})
    # new_axes[0].set_title('Enemy OPS: PLATE_Z × VAA', fontsize=13, fontweight='bold')
    # new_axes[0].set_ylabel('Pitch Height')
    # new_axes[0].set_xlabel('VAA Group')

    pivot_slg = final_grouped.pivot(index='z_group', columns='vaa_group', values='SLG')
    pivot_slg = pivot_slg.reindex(['Above', 'High Zone', 'Low Zone', 'Below'])

    sns.heatmap(pivot_slg, annot=True, fmt='.3f', cmap='RdYlBu', ax=new_axes[1],
                vmin=0.1, vmax=0.6, linewidths=1, linecolor='white',
                cbar_kws={'label': 'SLG'})
    new_axes[1].set_title('Enemy SLG: PLATE_Z × VAA', fontsize=13, fontweight='bold')
    new_axes[1].set_ylabel('Pitch Height')
    new_axes[1].set_xlabel('VAA Group')

    new_fig.suptitle('Four-Seam Fastball: Enemy OPS & SLG by VAA × Pitch Height',
                     fontsize=16, fontweight='bold')
    new_fig.tight_layout()

    # ----------------------------------------------------
    plt.show()

    # ── 수치 요약 ──
    print("=" * 70)
    print("VAA × PLATE_Z 조합별 Whiff Rate")
    print("=" * 70)
    for zone in zones:
        print(f"\n  [{zone}]")
        sub = grouped[grouped['z_group'] == zone]
        for _, row in sub.iterrows():
            wr = f"{row['whiff_rate']:.1%}" if pd.notna(row['whiff_rate']) else "N/A"
            print(f"    {row['vaa_group']:>8}: {wr}  "
                  f"({int(row['whiffs'])}/{int(row['swings'])} swings, n={int(row['n'])})")

    print("=" * 70)
    print("VAA × PLATE_Z 조합별 SLG")
    print("=" * 70)
    for zone in zones:
        print(f"\n  [{zone}]")
        sub = final_grouped[final_grouped['z_group'] == zone]
        for _, row in sub.iterrows():
            wr = f"{row['SLG']:.3}" if pd.notna(row['SLG']) else "N/A"
            print(f"    {row['vaa_group']:>8}: {wr}  "
                  f"({int(row['tb'])}, n={int(row['ab'])})")


if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="1234", dsn="localhost:1521/xe")
        cursor = conn.cursor()

        # sql = """
        #     SELECT *
        #     FROM TMP_TABLE
        #     WHERE PITCHER = 695243
        # """
          
        sql = """
            SELECT *
            FROM TMP_TABLE
        """

        cursor.execute(sql)
        raw = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
        df = raw.dropna(subset=FEATURES).copy()
        df = df.sort_values(
            by=["GAME_PK", "AT_BAT_NUMBER", "PITCH_NUMBER"],
            ascending=[True, True, True]
        )

        
        if len(df) > 0:
            plot_vaa_platez_whiff(df)
            #FF_analysis(df)
            #FF_Spin_Axis(df)        
        else:
            print("NO DATA FOUND")

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()