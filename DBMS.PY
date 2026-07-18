# import oracledb as odb 
# import pandas as pd 

# from pybaseball import batting_stats 
# from pybaseball import playerid_reverse_lookup

# conn = odb.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
# cursor = conn.cursor()

# batter_df = batting_stats(2025, qual=100)
# IDfg_list = batter_df['IDfg'].unique().tolist()
# id_lookup = playerid_reverse_lookup(player_ids=IDfg_list, key_type='fangraphs')

# batter_df = pd.merge(
#     left= batter_df,
#     right= id_lookup,
#     how='inner',
#     left_on='IDfg',
#     right_on='key_fangraphs'
# ).drop(columns=['key_fangraphs'])


# a = []
# sql = """
#     INSERT INTO BATTER_RECORD_INFORMATION
#         VALUES(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10,
#                 :11, :12, :13, :14, :15, :16, :17, :18, :19, :20,
#                 :21, :22, :23, :24, :25, :26, :27, :28, :29, :30,
#                 :31, :32, :33, :34, :35, :36, :37)
# """

# def f(type, data, digit = 0):
#     if type == 'str':
#         d = str(data) if pd.notna(data) else None 
    
#     elif type == 'int':
#         d = int(data) if pd.notna(data) else None

#     elif type == 'float':
#         d = float(round(data, digit)) if pd.notna(data) else None
#     a.append(d)

# for _, row in batter_df.iterrows():
#     f('int', row['key_mlbam'])
#     f('int', row['G'])
#     f('int', row['AB'])
#     f('int', row['PA'])
#     f('int', row['H'])
#     f('int', row['1B'])
#     f('int', row['2B'])
#     f('int', row['3B'])
#     f('int', row['HR'])
#     f('int', row['R'])
#     f('int', row['RBI'])
#     f('int', row['BB'])
#     f('int', row['IBB'])
#     f('int', row['SF'])
#     f('int', row['SH'])
#     f('int', row['GDP'])
#     f('int', row['SB'])
#     f('int', row['CS'])

#     f('float', row['AVG'], 3)
#     f('float', row['BB%'], 3)
#     f('float', row['K%'], 3)
#     f('float', row['BB/K'], 2)
    
#     f('float', row['OBP'], 3)
#     f('float', row['SLG'], 3)
#     f('float', row['OPS'], 3)
#     f('float', row['ISO'], 3)
#     f('float', row['BABIP'], 3)

#     f('float', row['GB/FB'], 2)
#     f('float', row['LD%'], 3)
#     f('float', row['GB%'], 3)
#     f('float', row['FB%'], 3)
#     f('float', row['IFFB%'], 3)
#     f('float', row['HR/FB'], 2)
#     f('float', row['IFH%'], 3)
#     f('float', row['BUH%'], 3)

#     f('float', row['WAR'], 2)
#     f('float', row['wRC+'], 1)

#     cursor.execute(sql, a)
  
#     a.clear()

# conn.commit()
# cursor.close()
# conn.close()




# from pybaseball import statcast
# import pandas as pd 
# import numpy as np 
# import oracledb as odb 

# from sqlalchemy import types

# from pybaseball import playerid_reverse_lookup

# conn = odb.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
# cursor = conn.cursor()

# sql = """
#     SELECT UNIQUE(BATTER) BATTER, 
#     MAX(CASE 
#         WHEN INNING_TOPBOT = 'Top' THEN AWAY_TEAM
#         ELSE HOME_TEAM
#     END) "TEAM_NAME"    
#     FROM TMP_TABLE
#     GROUP BY BATTER
#     ORDER BY "TEAM_NAME"
# """
# cursor.execute(sql)

# batter_ids = pd.DataFrame(data = cursor.fetchall(), columns= [row[0] for row in cursor.description])
# id_list = batter_ids['BATTER'].astype(int).tolist()

# player_lookup = playerid_reverse_lookup(id_list, key_type="mlbam")
# player_lookup['full_name'] = player_lookup['name_first'] + ' ' + player_lookup['name_last']
# player_lookup = pd.merge(
#     left = batter_ids,
#     right= player_lookup[['key_mlbam', 'full_name']],
#     how = 'inner',
#     left_on= 'BATTER',
#     right_on= 'key_mlbam'  
# ).drop(columns=['key_mlbam'])

# sql = """
#     insert into player
#     values(:1, :2, :3, :4)
# """
# for _, row in player_lookup.iterrows():
#     b_list = [int(row['BATTER']), str(row['full_name']), 'BATTER', str(row['TEAM_NAME'])]
#     cursor.execute(sql, b_list)

# conn.commit()
# cursor.close()
# conn.close()

# ============================================================================

import oracledb as ODB
import pandas as pd 
import numpy as np 

from pybaseball import statcast
from pybaseball import playerid_lookup
from pybaseball import playerid_reverse_lookup
from datetime import datetime    

# ================================
# 1. 데이터 수집
# ================================

print("Downloading Statcast data...")
df_first = statcast("2025-03-18", "2025-09-28")
print(f"Raw data size: {len(df_first)}")

# ================================
# 2. 데이터 정제
# ================================

#intercept_ball_minus_batter_pos_x_inches
#intercept_ball_minus_batter_pos_y_inches
# =>> 씨이이이발 이게 배트 중앙으로 부터 떨어진 거리였어 
normal_pitches = df_first.dropna(subset=['plate_x', 'plate_z', 'stand'])
violation_terms = ['violation', 'automatic_ball', 'automatic_strike']
violations = df_first[df_first['description'].str.contains('|'.join(violation_terms), na=False, case=False)]

df = pd.concat([normal_pitches, violations]).drop_duplicates()

#df = df_first[(df_first['events'].isin(non_pitch_events)) | (df_first['description'].isin (non_pitch_descriptions))]
df = df[df['game_type'] == 'R']
print(f"After cleaning: {len(df)}")

# DB 연결
conn = ODB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
cursor = conn.cursor()

sql = """ 
     INSERT INTO TMP_TABLE
         VALUES(:1, :2, :3, :4, :5, :6, :7, :8, :9, :10,
         :11, :12, :13, :14, :15, :16, :17, :18, :19, :20,
         :21, :22, :23, :24, :25, :26, :27, :28, :29, :30,
         :31, :32, :33, :34, :35, :36, :37, :38, :39, :40,
         :41, :42, :43, :44, :45, :46, :47, :48, :49, :50,
         :51, :52, :53, :54, :55, :56, :57, :58, :59, :60,
         :61, :62, :63, :64, :65, :66  
        )
"""
ele_list = []

def f(type, data, digit = 0):
    if type == 'str':
        d = str(data) if pd.notna(data) else None 
    
    elif type == 'int':
        d = int(data) if pd.notna(data) else None

    elif type == 'float':
        d = float(round(data, digit)) if pd.notna(data) else None
    ele_list.append(d)


for dd in df.itertuples():
    pitch_type = str(dd.pitch_type) if pd.notna(dd.pitch_type) else None
    ele_list.append(pitch_type)

    release_speed = float(dd.release_speed) if pd.notna(dd.release_speed) else None
    ele_list.append(release_speed)

    release_pos_x = float(dd.release_pos_x) if pd.notna(dd.release_pos_x) else None 
    ele_list.append(release_pos_x)

    release_pos_y = float(dd.release_pos_y) if pd.notna(dd.release_pos_y) else None
    ele_list.append(release_pos_y)

    release_pos_z = float(dd.release_pos_z) if pd.notna(dd.release_pos_z) else None
    ele_list.append(release_pos_z)

    release_spin_rate = int(dd.release_spin_rate) if pd.notna(dd.release_spin_rate) else None
    ele_list.append(release_spin_rate)

    pfx_x = float(dd.pfx_x) if pd.notna(dd.pfx_x) else None 
    ele_list.append(pfx_x)

    pfx_z = float(dd.pfx_z) if pd.notna(dd.pfx_z) else None
    ele_list.append(pfx_z)

    plate_x = float(dd.plate_x) if pd.notna(dd.plate_x) else None
    ele_list.append(plate_x)

    plate_z = float(dd.plate_z) if pd.notna(dd.plate_z) else None
    ele_list.append(plate_z) 

    vx0 = float(dd.vx0) if pd.notna(dd.vx0) else None
    ele_list.append(vx0)

    vy0 = float(dd.vy0) if pd.notna(dd.vy0) else None
    ele_list.append(vy0)

    vz0 = float(dd.vz0) if pd.notna(dd.vz0) else None
    ele_list.append(vz0)

    ax = float(dd.ax) if pd.notna(dd.ax) else None
    ele_list.append(ax)

    ay = float(dd.ay) if pd.notna(dd.ay) else None
    ele_list.append(ay)

    az = float(dd.az) if pd.notna(dd.az) else None
    ele_list.append(az)

    spin_axis = int(dd.spin_axis) if pd.notna(dd.spin_axis) else None
    ele_list.append(spin_axis)

    batter = int(dd.batter)
    ele_list.append(batter)

    pitcher = int(dd.pitcher)
    ele_list.append(pitcher) 

    game_pk = int(dd.game_pk) if pd.notna(dd.game_pk) else None 
    ele_list.append(game_pk)

    game_date = dd.game_date.to_pydatetime()
    ele_list.append(game_date)

    strikes = int(dd.strikes) if pd.notna(dd.strikes) else None
    ele_list.append(strikes) 

    balls = int(dd.balls) if pd.notna(dd.balls) else None 
    ele_list.append(balls) 

    outs_when_up = int(dd.outs_when_up) if pd.notna(dd.outs_when_up) else None
    ele_list.append(outs_when_up) 

    on_1b = int(dd.on_1b) if pd.notna(dd.on_1b) else None 
    ele_list.append(on_1b) 

    on_2b = int(dd.on_2b) if pd.notna(dd.on_2b) else None 
    ele_list.append(on_2b)

    on_3b = int(dd.on_3b) if pd.notna(dd.on_3b) else None 
    ele_list.append(on_3b) 

    at_bat_number = int(dd.at_bat_number) if pd.notna(dd.at_bat_number) else None 
    ele_list.append(at_bat_number) 

    pitch_number = int(dd.pitch_number) if pd.notna(dd.pitch_number) else None 
    ele_list.append(pitch_number) 

    description = str(dd.description) if pd.notna(dd.description) else None
    ele_list.append(description) 

    events = str(dd.events) if pd.notna(dd.events) else None
    ele_list.append(events) 

    des = str(dd.des) if pd.notna(dd.des) else None
    ele_list.append(des)  

    f('str', dd.stand)
    f('str', dd.p_throws) 
    f('str', dd.home_team)
    f('str', dd.away_team)
    f('str', dd.type)
    f('int',dd.hit_location)
    f('str', dd.bb_type)
    f('int', dd.inning)
    f('str', dd.inning_topbot)
    f('int', dd.hc_x)
    f('int', dd.hc_y)  
    f('int', dd.hit_distance_sc)
    f('float', dd.launch_speed, 1)
    f('int', dd.launch_angle)
    f('float', dd.estimated_ba_using_speedangle, 3)
    f('float', dd.estimated_woba_using_speedangle, 3)
    f('float', dd.woba_value, 3)
    f('int', dd.woba_denom)
    f('float', dd.babip_value, 3)
    f('int', dd.launch_speed_angle)
    f('int', dd.home_score)
    f('int', dd.post_home_score)
    f('int', dd.post_away_score)
    f('int', dd.away_score)
    f('float', dd.bat_speed, 1)
    f('float', dd.swing_length, 1)
    f('float', dd.estimated_slg_using_speedangle, 2)
    f('float', dd.home_win_exp, 4)
    f('float', dd.arm_angle, 1)
    f('float', dd.attack_angle, 1)
    f('float', dd.attack_direction, 1)
    f('float', dd.intercept_ball_minus_batter_pos_x_inches, 3)
    f('float', dd.intercept_ball_minus_batter_pos_y_inches, 3)
    f('float', dd.swing_path_tilt, 2)

    cursor.execute(sql, ele_list)
    ele_list.clear()

conn.commit()
cursor.close()
conn.close()