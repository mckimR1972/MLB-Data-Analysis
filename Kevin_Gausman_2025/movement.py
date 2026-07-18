import oracledb as DB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, LogisticRegression
# from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_PITCH_COLORS = {
    "FF": "#BE0B0B", "SI": "#7F77DD", "FC": "#d4a017",
    "SL": "#EC7DB1", "SV": "#d03030", "ST": "#c02828",
    "CU": "#40b040", "KC": "#38a038", "CH": "#4090d0",
    "FS": "#1DD1D1", "KN": "#8060b0",
}

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

    tracking = 0 
    for i in range(0, 99):
        tracking += (np.sqrt((real_x[i] - real_x[i + 1]) ** 2 + 
                             (y[i] - y[i + 1]) ** 2))

    tracking -= np.sqrt((real_x[0] - target_x) ** 2 + (y[0] - y[-1]) ** 2)
    tmp_dict['Z_FOR_Y25'].append(z25)
    tmp_dict['X_FOR_Y25'].append(x25)
    tmp_dict['X_FOR_Y15'].append(x15)
    tmp_dict['Z_FOR_Y15'].append(z15)
    tmp_dict['X_BENT_VALUE'].append(tracking)
    
    
def analysis(df: pd.DataFrame):
    # 1. 데이터 전처리 로직 (동일)
    df = df.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER']).reset_index(drop=True)
    tmp_dict = {'X_FOR_Y25':[], 'Z_FOR_Y25':[], 'X_FOR_Y15':[], 'Z_FOR_Y15':[], 'X_BENT_VALUE':[]}

    for _, row in df.iterrows():
        pitch_tunnel_cal(row, tmp_dict)

    tmp_df = pd.DataFrame(tmp_dict)
    df = df.reset_index(drop=True) 
    df = pd.concat([df, tmp_df], axis=1)

    # 파생 변수 생성 생략 (이전 코드와 완전히 동일)
    df['DROP'] = df['PLATE_Z'] - df['Z_FOR_Y25']
    df['MOVE'] = -df['PLATE_X'] + df['X_FOR_Y25']
    df['DROP_RATE'] =  df['DROP'] / (-df['RELEASE_POS_Z'] + df['PLATE_Z'])
    df['MOVE_RATE'] = df['MOVE'] / (-df['RELEASE_POS_X'] + df['PLATE_X'])
    df['DID_SWING'] = np.where(
        df['PITCH_DESCRIPTION'].isin(['hit_into_play', 'foul', 'swinging_strike', 'swinging_strike_blocked', 
                                      'foul_tip', 'foul_bunt_tip', 'bunt']), 
                                      1, 0
    )
    df['MISSED_SWING'] = np.where(
        df['PITCH_DESCRIPTION'].isin(['swinging_strike', 'swinging_strike_blocked']) , 1, 0
    )
    df['DID_HIT'] = np.where(
        df['PITCH_EVENTS'].isin(['single', 'double', 'triple', 'home_run']), 1, 0
    )

    df_color = df['PITCH_TYPE'].map(DEFAULT_PITCH_COLORS)

    l = df[df['STAND'] == 'L']
    l_color = l['PITCH_TYPE'].map(DEFAULT_PITCH_COLORS)
    
    r = df[df['STAND'] == 'R']
    r_color = r['PITCH_TYPE'].map(DEFAULT_PITCH_COLORS)

    def get_border_line():
        border_line = plt.Rectangle(
            [-0.83, sz_bot], 1.66, sz_top - sz_bot, color='black', linewidth=1.5, fill=False
        )
        return border_line 
    

    # # ================================================
    # # 포심 분석 
    # # ================================================
    ff_df = df[df['PITCH_TYPE'] == 'FF']
    
    l,r  = ff_df[ff_df['STAND'] == 'L'], ff_df[ff_df['STAND'] == 'R']
    
    fig_ff, ax_ff = plt.subplots(1, 2, figsize=(16, 8))
    l_plot = sns.scatterplot(
        data=ff_df, x='PLATE_X', y='PFX_X', s= 15, color = DEFAULT_PITCH_COLORS['FF'], ax=ax_ff[0]
    )
    print(ff_df[['PLATE_X', 'PFX_X', 'PFX_Z', 'SPIN_AXIS', 'ARM_ANGLE']].corr())
    # l_plot.set_xlim(-2.5,2.5)
    # l_plot.set_ylim(0, 5)
    # l_plot.add_patch(get_border_line())

    r_plot = sns.scatterplot(
        data=r, x='PLATE_X', y='PLATE_Z', s= 15, color = DEFAULT_PITCH_COLORS['FF'], ax=ax_ff[1]
    )
    r_plot.set_xlim(-2.5,2.5)
    r_plot.set_ylim(0, 5)
    r_plot.add_patch(get_border_line())

    l_plot.set_title(f"VS Left Batter : Four seam Drop Rate and Position at Y = 25")
    r_plot.set_title(f"VS Right Batter : Four seam Drop Rate and Position at Y = 25")
    #plt.show()

    # ================================================
    # 슬라이더 + 스플리터 분석 => 둘이 궤적이나 무브먼트가 거의 유사해 보임 
    # ================================================
    # sl_df = df[df['PITCH_TYPE'] == 'SL']
    # tmp_df = df[df['PITCH_TYPE'].isin(['SL', 'FS'])]
    # colors = tmp_df['PITCH_TYPE'].map(DEFAULT_PITCH_COLORS)
    # fig_sl, ax_sl = plt.subplots(1, 1, figsize=(9, 6))
    # plot_sl = sns.scatterplot(
    #     data=tmp_df, x='X_FOR_Y25', y='MOVE_RATE', s=15, hue ='PITCH_TYPE'
    # )
    # plot_sl.set_xlim(2, 6)
    # plot_sl.set_ylim(0.5, 1)


    # plot_sl.set_title(f"Slider Movement Rate And Drop Rate")
    # rate = round(len(sl_df[sl_df['DID_SWING'] == 1]) / len(sl_df) * 100 , 1)
    # print('='*50)
    # print(f"슬라이더 스윙 유도율 : {rate} %")
    # whiff_rate = round(len(sl_df[sl_df['MISSED_SWING'] == 1]) / len(sl_df) * 100 , 1)
    # print(f"슬라이더 헛스윙 유도율 : {whiff_rate} %")
    # print(f"슬라이더 스윙 유도 대비 헛스윙 비율 : {round(whiff_rate / rate * 100)} %")
    
    #plt.show()
    # ================================================
    # 스플리터 분석 
    # ================================================ 
    # fs_df = df[df['PITCH_TYPE'].isin(['FS', 'FF'])]
    # colors = fs_df['PITCH_TYPE'].map(DEFAULT_PITCH_COLORS)
    # fig2, ax2 = plt.subplots(1, 1, figsize = (12, 8))
    # plot_fs = sns.scatterplot(
    #     data=fs_df, x='Z_FOR_Y25', y='DROP_RATE', color = colors, s=15
    # )
    # plot_fs.set_xlim(2, 6)
    # plot_fs.set_ylim(0.5, 1)
    # rate = round(len(fs_df[fs_df['DID_SWING'] == 1]) / len(fs_df) * 100 , 1)
    # print('=' * 50)
    # print(f"스플리터 스윙 유도율 : {rate} %")
    # whiff_rate = round(len(fs_df[fs_df['MISSED_SWING'] == 1]) / len(fs_df) * 100 , 1)
    # print(f"스플리터 헛스윙 유도율 : {whiff_rate} %")
    # print(f"스플리터 스윙 유도 대비 헛스윙 비율 : {round(whiff_rate / rate * 100)} %")
    # plot2.add_patch(get_border_line())
    # plot2.set_xlim(-2.5, 2.5)
    # plot2.set_ylim(-1, 5)
    # plot2.set_box_aspect(5/6)
    # plot_fs.set_title("Splitter Drop Rate And Position at Y = 25")
    plt.show()

    
if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
        cursor = conn.cursor()
        
        #592332
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
                    'PFX_X', 'PFX_Z', 'RELEASE_SPIN_RATE', 'PITCH_TYPE'
                ]
        )
        
        analysis(clean_df)


    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'conn' in locals(): conn.close()

