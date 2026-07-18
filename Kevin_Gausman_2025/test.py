import oracledb as DB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

def predict_hit_distance(df:pd.DataFrame):
    X = df[['ATTACK_ANGLE', 'ATTACK_DIRECTION', 
                    'INTERCEPT_X', 'INTERCEPT_Y', 'SWING_PATH_TILT']]
    y = df[['LAUNCH_ANGLE']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    print(f"R2 Score: {r2_score(y_test, y_pred)}")

def pitch_tunnel_cal(row: pd.Series, tmp_dict : dict):
    target_x, target_z = row['PLATE_X'], row['PLATE_Z']
    # 물리 궤적 계산 
    # 피칭 터널 시작 - 포구지점 
    sqrt = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 1.417)) # home_plate_y = 1.417 
    
    end_t = (- row['VY0'] - sqrt) / row['AY'] 
    t = np.linspace(0, end_t, 100)

    y = row['RELEASE_POS_Y'] + row['VY0']*t + .5*row['AY']*t**2
    t = t[y >= 0]

    theorem_x = row['RELEASE_POS_X'] + row['VX0']*t + .5*row['AX']*t**2
    y = row['RELEASE_POS_Y'] + row['VY0']*t + .5*row['AY']*t**2
    theorem_z = row['RELEASE_POS_Z'] + row['VZ0']*t + .5*row['AZ']*t**2
    error_x = target_x - theorem_x[-1]
    error_z = target_z - theorem_z[-1]

    # 비행 시간의 비율(0~1)에 따라 오차를 점진적으로 배분
    correction_factor = (t / end_t)

    # 25피트 지점에서 z좌표 위치 
    sqrt25 = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 25))
    t25 = (-row['VY0'] - sqrt25) / row['AY']
    
    # Y좌표가 25feet일 때 t값으로 그 때 Z좌표 구하기 
    z25 = row['RELEASE_POS_Z'] + row['VZ0'] * t25 + 0.5*row['AZ']*t25**2
    z25 = z25 + (error_z * t25 / end_t)
    x25 = row['RELEASE_POS_X'] + row['VX0'] * t25 + 0.5*row['AX']*t25**2
    x25 = x25 + (error_x * t25 / end_t)

    # 15피트 지점에서 z좌표 위치 
    sqrt15 = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 15))
    t15 = (-row['VY0'] - sqrt15) / row['AY']
    
    # Y좌표가 15feet일 때 t값으로 그 때 Z좌표 구하기 
    z15 = row['RELEASE_POS_Z'] + row['VZ0'] * t15 + 0.5*row['AZ']*t15**2
    z15 = z15 + (error_z * t15 / end_t)
    x15 = row['RELEASE_POS_X'] + row['VX0'] * t15 + 0.5*row['AX']*t15**2
    x15 = x15 + (error_x * t15 / end_t)

    
    real_z = theorem_z + (error_z * correction_factor)   
    real_x = theorem_x + (error_x * correction_factor)

    tmp_dict['Z_FOR_Y25'].append(z25)
    tmp_dict['X_FOR_Y25'].append(x25)
    tmp_dict['X_FOR_Y15'].append(x15)
    tmp_dict['Z_FOR_Y15'].append(z15)


import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines # 커스텀 범례를 위한 모듈 추가

# (기존 pitch_tunnel_cal 함수 생략)

def analysis(df: pd.DataFrame, plot_limit=20):
    # 1. 데이터 전처리 로직 (동일)
    df = df.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    tmp_dict = {'X_FOR_Y25':[], 'Z_FOR_Y25':[], 'X_FOR_Y15':[], 'Z_FOR_Y15':[]}

    for _, row in df.iterrows():
        pitch_tunnel_cal(row, tmp_dict)

    tmp_df = pd.DataFrame(tmp_dict)
    df = df.reset_index(drop=True) 
    df = pd.concat([df, tmp_df], axis=1)

    # 파생 변수 생성 생략 (이전 코드와 완전히 동일)
    df['DROP'] = df['PLATE_Z'] - df['Z_FOR_Y15']
    df['MOVE'] = df['PLATE_X'] - df['X_FOR_Y15']
    
    # ---------------------------------------------------------
    # 2. 대시보드 화면 분할 및 플롯 그리기
    # ---------------------------------------------------------
    pitch_colors = {
        'FF': 'red', 'SL': 'gold', 'CH': 'green', 'CU': 'blue', 
        'SI': 'darkorange', 'FC': 'brown', 'FS': 'cyan', 'ST': 'purple'
    }
    
    pt_col = 'pitch_type' if 'pitch_type' in df.columns else 'PITCH_TYPE'
    desc_col = 'description' if 'description' in df.columns else 'PITCH_DESCRIPTION'
    velo_col = 'release_speed' if 'release_speed' in df.columns else 'RELEASE_SPEED'
    balls_col = 'balls' if 'balls' in df.columns else 'BALLS'
    strikes_col = 'strikes' if 'strikes' in df.columns else 'STRIKES'

    at_bat_groups = df.groupby(['GAME_PK', 'AT_BAT_NUMBER'])
    
    count = 0
    for (game, ab_num), group in at_bat_groups:
        if count >= plot_limit:
            break
            
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        ax_loc = fig.add_subplot(gs[0, 0])
        ax_mov = fig.add_subplot(gs[0, 1])
        ax_tab = fig.add_subplot(gs[1, :])
        
        # ==========================================
        # [상단 좌측] 투구 로케이션 (ax_loc) - 이전과 동일
        # ==========================================
        top = group['sz_top'].mean() if 'sz_top' in group.columns else 3.5
        bot = group['sz_bot'].mean() if 'sz_bot' in group.columns else 1.5
        zone = patches.Rectangle((-0.83, bot), 1.66, top - bot, linewidth=2, edgecolor='black', facecolor='none')
        ax_loc.add_patch(zone)
        
        for _, row in group.iterrows():
            p_type = row[pt_col] if pd.notna(row.get(pt_col)) else 'UN'
            color = pitch_colors.get(p_type, 'gray')
            ax_loc.scatter(row['PLATE_X'], row['PLATE_Z'], s=200, c=color, alpha=0.8, edgecolors='black')
            ax_loc.text(row['PLATE_X'], row['PLATE_Z'], str(int(row['PITCH_NUMBER'])),
                     fontsize=10, ha='center', va='center', color='black', fontweight='bold')
            
        ax_loc.set_xlim(-2.5, 2.5)
        ax_loc.set_ylim(0, 5)
        ax_loc.set_aspect('equal')
        ax_loc.set_title('Pitch Location')
        ax_loc.grid(True, linestyle='--', alpha=0.5)

        batter_stand = group['STAND'].iloc[0] # 해당 타석의 첫 번째 투구 기록에서 타자 스탠스 추출
            
        if batter_stand == 'R': # 우타자 (포수 시점 좌측)
            ax_loc.axvline(x=-2.0, color='gray', linestyle='-', linewidth=6, alpha=0.4)
            ax_loc.text(-2.15, 2.5, 'Right-Handed Batter', rotation=90, va='center', ha='center', color='gray', fontweight='bold')
        elif batter_stand == 'L': # 좌타자 (포수 시점 우측)
            ax_loc.axvline(x=2.0, color='gray', linestyle='-', linewidth=6, alpha=0.4)
            ax_loc.text(2.15, 2.5, 'Left-Handed Batter', rotation=-90, va='center', ha='center', color='gray', fontweight='bold')

        # ==========================================
        # [상단 우측] 투구 무브먼트 (구종 색상 연동)
        # ==========================================
        pitch_nums = group['PITCH_NUMBER'].astype(int).values
        moves = group['MOVE'].values
        drops = group['DROP'].values
        
        # 각 투구의 구종 색상을 리스트로 추출
        colors = [pitch_colors.get(row.get(pt_col, 'UN'), 'gray') for _, row in group.iterrows()]
        
        # 투구 순서대로 점과 선분을 하나씩 그리기
        for i in range(len(pitch_nums)):
            # 1. 현재 투구의 '점(Marker)' 찍기
            # MOVE는 원형(o), DROP은 사각형(s)
            ax_mov.plot(pitch_nums[i], moves[i], marker='o', color=colors[i], markersize=8)
            ax_mov.plot(pitch_nums[i], drops[i], marker='s', color=colors[i], markersize=8)
            
            # 2. 두 번째 투구부터 이전 투구와 연결하는 '선(Line)' 긋기
            if i > 0:
                prev_x, curr_x = pitch_nums[i-1], pitch_nums[i]
                
                # 선의 색상은 '현재 투구(colors[i])'의 색상으로 지정
                # MOVE는 실선(-), DROP은 점선(--)
                ax_mov.plot([prev_x, curr_x], [moves[i-1], moves[i]], color=colors[i], linewidth=2, linestyle='-')
                ax_mov.plot([prev_x, curr_x], [drops[i-1], drops[i]], color=colors[i], linewidth=2, linestyle='-')
        
        ax_mov.axhline(0, color='black', linewidth=1, linestyle='-')
        ax_mov.set_xticks(pitch_nums)
        ax_mov.set_title('Movement by Pitch Type (Color = Current Pitch)')
        
        # 커스텀 범례 생성 (색상이 다양해졌으므로 선 모양으로만 MOVE/DROP 구분)
        legend_move = mlines.Line2D([], [], color='black', marker='o', linestyle='-', label='MOVE (X)')
        legend_drop = mlines.Line2D([], [], color='black', marker='s', linestyle='-', label='DROP (Z)')
        ax_mov.legend(handles=[legend_move, legend_drop])
        ax_mov.grid(True, linestyle=':', alpha=0.6)

        # ==========================================
        # [하단] 투구 내역 표 (ax_tab) - 이전과 동일
        # ==========================================
        ax_tab.axis('off')
        table_data = []
        for _, row in group.iterrows():
            p_num = int(row['PITCH_NUMBER'])
            p_type = row.get(pt_col, '-')
            velo = round(row.get(velo_col, 0), 1) if pd.notna(row.get(velo_col)) else '-'
            desc = row.get(desc_col, '-')
            b = int(row.get(balls_col, 0)) if pd.notna(row.get(balls_col)) else 0
            s = int(row.get(strikes_col, 0)) if pd.notna(row.get(strikes_col)) else 0
            table_data.append([p_num, p_type, f"{velo} mph", desc, f"{b}-{s}"])
            
        col_labels = ['Pitch No', 'Pitch Type', 'Velocity', 'Description', 'Count (B-S)']
        table = ax_tab.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax_tab.set_title(f"At-Bat Summary (Game: {game} / AB: {ab_num})", fontweight="bold")

        plt.tight_layout()
        plt.show()
        
        count += 1
        
    return df

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
        cursor = conn.cursor()
        
        # 데이터 로드
        sql = """
            SELECT * 
            FROM TMP_TABLE 
            WHERE 
                PITCHER = 592332
        """

        cursor.execute(sql)
        raw_df = pd.DataFrame(cursor.fetchall(), columns=[d[0] for d in cursor.description])
        
        cursor.execute(f"SELECT AVG(SZ_TOP), AVG(SZ_BOT) FROM batter_strike_zone")
        sz_top, sz_bot = cursor.fetchone()
        
        # 전처리 
        clean_df = raw_df.dropna(
            subset=['PLATE_X', 'PLATE_Z', 
                    'RELEASE_POS_X', 'RELEASE_POS_Y', 'RELEASE_POS_Z',
                    'VX0', 'VY0', 'VZ0', 
                    'AX', 'AY', 'AZ', 
                    'PFX_X', 'PFX_Z', 'RELEASE_SPIN_RATE'
                ]
        )
        #print(len(clean_df))
        
        analysis(clean_df)


    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'conn' in locals(): conn.close()