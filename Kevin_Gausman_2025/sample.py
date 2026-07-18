import oracledb as DB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import trim_mean
plt.rcParams['font.family'] = 'Malgun Gothic'

DEFAULT_PITCH_COLORS = {
    "FF": "#BE0B0B", "SI": "#7F77DD", "FC": "#d4a017",
    "SL": "#EC7DB1", "SV": "#d03030", "ST": "#c02828",
    "CU": "#40b040", "KC": "#38a038", "CH": "#4090d0",
    "FS": "#1DD1D1", "KN": "#8060b0",
}


def pitch_tunnel_cal(row: pd.Series, tmp_dict: dict):
    target_x, target_z = row['PLATE_X'], row['PLATE_Z']
    sqrt = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 1.417))
    end_t = (-row['VY0'] - sqrt) / row['AY']
    t = np.linspace(0, end_t, 100)

    y = row['RELEASE_POS_Y'] + row['VY0'] * t + .5 * row['AY'] * t ** 2
    t = t[y >= 0]

    theorem_x = row['RELEASE_POS_X'] + row['VX0'] * t + .5 * row['AX'] * t ** 2
    y = row['RELEASE_POS_Y'] + row['VY0'] * t + .5 * row['AY'] * t ** 2
    theorem_z = row['RELEASE_POS_Z'] + row['VZ0'] * t + .5 * row['AZ'] * t ** 2
    error_x = target_x - theorem_x[-1]
    error_z = target_z - theorem_z[-1]

    correction_factor = (t / end_t)

    # y=25 지점
    sqrt25 = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 25))
    t25 = (-row['VY0'] - sqrt25) / row['AY']
    z25 = row['RELEASE_POS_Z'] + row['VZ0'] * t25 + 0.5 * row['AZ'] * t25 ** 2
    z25 = z25 + (error_z * t25 / end_t)
    x25 = row['RELEASE_POS_X'] + row['VX0'] * t25 + 0.5 * row['AX'] * t25 ** 2
    x25 = x25 + (error_x * t25 / end_t)

    # y=15 지점
    sqrt15 = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 15))
    t15 = (-row['VY0'] - sqrt15) / row['AY']
    z15 = row['RELEASE_POS_Z'] + row['VZ0'] * t15 + 0.5 * row['AZ'] * t15 ** 2
    z15 = z15 + (error_z * t15 / end_t)
    x15 = row['RELEASE_POS_X'] + row['VX0'] * t15 + 0.5 * row['AX'] * t15 ** 2
    x15 = x15 + (error_x * t15 / end_t)

    tmp_dict['Z_FOR_Y25'].append(z25)
    tmp_dict['X_FOR_Y25'].append(x25)
    tmp_dict['X_FOR_Y15'].append(x15)
    tmp_dict['Z_FOR_Y15'].append(z15)


def analysis(df: pd.DataFrame, sz_top: float, sz_bot: float):
    # ================================================
    # 1. 궤적 계산
    # ================================================
    df = df.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER']).reset_index(drop=True)
    tmp_dict = {'X_FOR_Y25': [], 'Z_FOR_Y25': [], 'X_FOR_Y15': [], 'Z_FOR_Y15': []}

    for _, row in df.iterrows():
        pitch_tunnel_cal(row, tmp_dict)

    tmp_df = pd.DataFrame(tmp_dict)
    df = pd.concat([df, tmp_df], axis=1)

    # 스윙 관련 파생 변수
    df['DID_SWING'] = np.where(
        df['PITCH_DESCRIPTION'].isin([
            'hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked',
            'foul_tip', 'foul_bunt_tip', 'bunt'
        ]), 1, 0
    )
    df['MISSED_SWING'] = np.where(
        df['PITCH_DESCRIPTION'].isin(['swinging_strike', 'swinging_strike_blocked']), 1, 0
    )

    # ================================================
    # 2. 포심 기준선 산출 (상하위 10% 절사평균)
    # ================================================
    ff_df = df[df['PITCH_TYPE'] == 'FF']

    if len(ff_df) < 10:
        print("포심 데이터가 부족합니다 (10구 미만)")
        return

    ff_x25_trimmed = trim_mean(ff_df['X_FOR_Y25'], proportiontocut=0.1)
    ff_z25_trimmed = trim_mean(ff_df['Z_FOR_Y25'], proportiontocut=0.1)

    # 포심의 y25 -> plate 구간 후반 낙차 절사평균 (기준선)
    ff_late_drop = ff_df['Z_FOR_Y25'] - ff_df['PLATE_Z']
    ff_late_drop_trimmed = trim_mean(ff_late_drop, proportiontocut=0.1)

    # 포심의 y25 -> plate 구간 수평 변화 절사평균 (기준선)
    ff_late_move = ff_df['X_FOR_Y25'] - ff_df['PLATE_X']
    ff_late_move_trimmed = trim_mean(ff_late_move, proportiontocut=0.1)

    print("=" * 60)
    print(f"[포심 기준선 - 절사평균 (상하위 10% 제거)]")
    print(f"  Y25 좌표  : x = {ff_x25_trimmed:.3f} ft, z = {ff_z25_trimmed:.3f} ft")
    print(f"  후반 낙차 : {ff_late_drop_trimmed:.3f} ft")
    print(f"  후반 수평 : {ff_late_move_trimmed:.3f} ft")
    print("=" * 60)

    # ================================================
    # 3. 스플리터 지표 계산
    # ================================================
    fs_df = df[df['PITCH_TYPE'] == 'FS'].copy()

    if len(fs_df) == 0:
        print("스플리터 데이터가 없습니다.")
        return

    # 터널 유사도: y25 지점에서 포심 절사평균 좌표와의 유클리드 거리
    # 값이 작을수록 포심과 구분이 안 됨 (터널링 좋음)
    fs_df['TUNNEL_DIST'] = np.sqrt(
        (fs_df['X_FOR_Y25'] - ff_x25_trimmed) ** 2 +
        (fs_df['Z_FOR_Y25'] - ff_z25_trimmed) ** 2
    )

    # 후반 분리도: y25 -> plate 구간에서 포심 대비 추가 낙차 (ft)
    # 값이 클수록 포심보다 후반에 더 많이 떨어짐
    fs_late_drop = fs_df['Z_FOR_Y25'] - fs_df['PLATE_Z']
    fs_df['LATE_BREAK_DIFF'] = fs_late_drop - ff_late_drop_trimmed

    # 후반 수평 분리도
    fs_late_move = fs_df['X_FOR_Y25'] - fs_df['PLATE_X']
    fs_df['LATE_MOVE_DIFF'] = fs_late_move - ff_late_move_trimmed

    # ================================================
    # 4. 요약 통계 출력
    # ================================================
    swing_count = fs_df['DID_SWING'].sum()
    whiff_count = fs_df['MISSED_SWING'].sum()
    total = len(fs_df)

    print(f"\n[스플리터 요약 (총 {total}구)]")
    print(f"  스윙 유도율        : {swing_count / total * 100:.1f}%")
    if swing_count > 0:
        print(f"  헛스윙률 (Whiff%)  : {whiff_count / swing_count * 100:.1f}%")
    print(f"  터널 거리 평균     : {fs_df['TUNNEL_DIST'].mean():.3f} ft")
    print(f"  후반 추가 낙차 평균: {fs_df['LATE_BREAK_DIFF'].mean():.3f} ft")
    print("=" * 60)

    # ================================================
    # 5. 시각화
    # ================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # --- Plot 1: 터널 유사도 vs 후반 분리도 (핵심 scatter) ---
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        fs_df['TUNNEL_DIST'],
        fs_df['LATE_BREAK_DIFF'],
        c=fs_df['MISSED_SWING'].map({0: '#AAAAAA', 1: '#FF4444'}),
        s=40, alpha=0.7, edgecolors='white', linewidth=0.5
    )
    ax1.set_xlabel('Tunnel Distance (ft)\n← 포심과 유사 | 포심과 다름 →', fontsize=11)
    ax1.set_ylabel('Late Break Diff (ft)\n↑ 포심 대비 추가 낙차 큼', fontsize=11)
    ax1.set_title('Splitter Effectiveness\nTunnel Quality vs Late Break', fontsize=13, fontweight='bold')
    ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # 범례
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', markersize=8, label='헛스윙'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#AAAAAA', markersize=8, label='그 외'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 좌상단 = 효과적인 영역 표시
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    ax1.fill_between(
        [xlim[0], np.median(fs_df['TUNNEL_DIST'])],
        ylim[1], np.median(fs_df['LATE_BREAK_DIFF']),
        alpha=0.08, color='green', label='효과적 영역'
    )
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)

    # --- Plot 2: y25 지점에서 포심 vs 스플리터 좌표 비교 ---
    ax2 = axes[1]
    ax2.scatter(
        ff_df['X_FOR_Y25'], ff_df['Z_FOR_Y25'],
        c=DEFAULT_PITCH_COLORS['FF'], s=15, alpha=0.4, label='FF (포심)'
    )
    ax2.scatter(
        fs_df['X_FOR_Y25'], fs_df['Z_FOR_Y25'],
        c=DEFAULT_PITCH_COLORS['FS'], s=25, alpha=0.7, label='FS (스플리터)'
    )
    # 포심 절사평균 기준점
    ax2.scatter(
        ff_x25_trimmed, ff_z25_trimmed,
        c='black', s=150, marker='X', zorder=5, label='FF 절사평균'
    )
    ax2.set_xlabel('X at Y=25 (ft)', fontsize=11)
    ax2.set_ylabel('Z at Y=25 (ft)', fontsize=11)
    ax2.set_title('Y=25 지점 좌표 분포\nFF vs FS Tunnel Overlap', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)

    # --- Plot 3: 터널 거리 구간별 헛스윙률 ---
    ax3 = axes[2]
    fs_df['TUNNEL_BIN'] = pd.qcut(
        fs_df['TUNNEL_DIST'], q=4, labels=['Q1\n(가장 유사)', 'Q2', 'Q3', 'Q4\n(가장 다름)'],
        duplicates='drop'
    )

    bin_stats = fs_df.groupby('TUNNEL_BIN', observed=True).agg(
        whiff_rate=('MISSED_SWING', 'mean'),
        swing_rate=('DID_SWING', 'mean'),
        count=('MISSED_SWING', 'count')
    ).reset_index()

    x_pos = range(len(bin_stats))
    bars_whiff = ax3.bar(
        x_pos, bin_stats['whiff_rate'] * 100,
        color=DEFAULT_PITCH_COLORS['FS'], alpha=0.8, label='헛스윙률'
    )
    ax3.bar(
        x_pos, bin_stats['swing_rate'] * 100,
        color=DEFAULT_PITCH_COLORS['FS'], alpha=0.25, label='스윙 유도율'
    )

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(bin_stats['TUNNEL_BIN'], fontsize=10)
    ax3.set_xlabel('터널 거리 구간 (4분위)', fontsize=11)
    ax3.set_ylabel('%', fontsize=11)
    ax3.set_title('터널 유사도 구간별\n스윙 유도율 & 헛스윙률', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)

    # 막대 위에 투구 수 표시
    for i, (bar, cnt) in enumerate(zip(bars_whiff, bin_stats['count'])):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'n={cnt}', ha='center', fontsize=9, color='gray'
        )

    plt.tight_layout()
    plt.show()

    return fs_df


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

        cursor.execute("SELECT AVG(SZ_TOP), AVG(SZ_BOT) FROM batter_strike_zone")
        sz_top, sz_bot = cursor.fetchone()

        clean_df = raw_df.dropna(
            subset=[
                'PLATE_X', 'PLATE_Z',
                'RELEASE_POS_X', 'RELEASE_POS_Y', 'RELEASE_POS_Z',
                'VX0', 'VY0', 'VZ0',
                'AX', 'AY', 'AZ',
                'PFX_X', 'PFX_Z', 'RELEASE_SPIN_RATE', 'PITCH_TYPE'
            ]
        )

        result_df = analysis(clean_df, sz_top, sz_bot)

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'conn' in locals():
            conn.close()