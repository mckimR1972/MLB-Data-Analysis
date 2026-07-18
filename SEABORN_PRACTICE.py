import oracledb as DB 
import seaborn as SNS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 

from sklearn.cluster import DBSCAN


def calculate_position_at_y(row, target_y=23.8):
    y0 = row['RELEASE_POS_Y']
    vy0 = row['VY0']
    ay = row['AY']

    delta = vy0 ** 2 - 2 * ay * (y0 - target_y)

    if delta < 0: return np.nan, np.nan 

    t = (- vy0 - np.sqrt(delta)) / ay 

    x_tunnel = row['RELEASE_POS_X'] + row['VX0'] * t + 0.5 * row['AX'] * (t ** 2) 
    z_tunnel = row['RELEASE_POS_Z'] + row['VZ0'] * t + 0.5 * row['AZ'] * (t ** 2) 

    return x_tunnel, z_tunnel

conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
cursor = conn.cursor() 

sql = """ 
    SELECT 
        PITCH_TYPE,
        RELEASE_SPEED,
        RELEASE_POS_X, RELEASE_POS_Y, RELEASE_POS_Z,
        VX0, VY0, VZ0,
        AX, AY, AZ,
        PLATE_X, PLATE_Z 
    FROM TMP_TABLE
    WHERE PITCHER = (
        SELECT PLAYER_ID
        FROM PLAYER
        WHERE PLAYER_NAME = 'Paul Skenes'
    )
    AND PITCH_TYPE IN ('FF', 'ST', 'FS')
    AND RELEASE_POS_Z IS NOT NULL
"""

cursor.execute(sql)
df_tunnel = pd.DataFrame(
    data= cursor.fetchall(), 
    columns=[row[0] for row in cursor.description]
)

df_origin = df_tunnel 

tunnel_coords = df_tunnel.apply(lambda row : calculate_position_at_y(row), axis = 1)

df_tunnel[['TUNNEL_X', 'TUNNEL_Z']] = pd.DataFrame(tunnel_coords.tolist(), index=df_tunnel.index)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# 구종별 색상 설정
palette = {'FF': 'red', 'ST': 'green', 'FS' : 'purple' }

# 1. 릴리스 포인트 (Release Point)
SNS.scatterplot(data=df_tunnel, x='RELEASE_POS_X', y='RELEASE_POS_Z', hue='PITCH_TYPE', palette=palette, alpha=0.3, ax=axes[0])
axes[0].set_title("1. Release Point (Start)")
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(0, 7)

# 2. 터널 포인트 (Tunnel Point - 23.8ft)
SNS.scatterplot(data=df_tunnel, x='TUNNEL_X', y='TUNNEL_Z', hue='PITCH_TYPE', palette=palette, alpha=0.3, ax=axes[1])
axes[1].set_title("2. Tunnel Point (Decision Time)")
# 여기에 원을 그려서 두 구종이 얼마나 겹치는지 보여주면 좋습니다.

# 3. 플레이트 (Plate Location - End)
SNS.scatterplot(data=df_tunnel, x='PLATE_X', y='PLATE_Z', hue='PITCH_TYPE', palette=palette, alpha=0.3, ax=axes[2])
axes[2].set_title("3. Plate Location (Result)")

# 스트라이크 존 추가 (각 그래프에)
for ax in axes:
    ax.add_patch(plt.Rectangle((-0.85, 1.5), 1.7, 2.0, fill=False, color='black', lw=2))
    ax.set_aspect('equal')


avg_pos_FF = [0, 0, 0]
avg_pos_CU = [0, 0, 0]

for d in df_origin.itertuples():
    if d.PITCH_TYPE == 'FF':
        avg_pos_FF[0] += d.RELEASE_POS_X 
        avg_pos_FF[1] += d.RELEASE_POS_Z 
        avg_pos_FF[2] += 1 

    else:
        avg_pos_CU[0] += d.RELEASE_POS_X 
        avg_pos_CU[1] += d.RELEASE_POS_Z 
        avg_pos_CU[2] += 1 

avg_ff_x = round(avg_pos_FF[0] / avg_pos_FF[2], 2)
avg_ff_z = round(avg_pos_FF[0] / avg_pos_FF[2], 2)
avg_cu_x = round(avg_pos_CU[0] / avg_pos_CU[2], 2)
avg_cu_z = round(avg_pos_CU[0] / avg_pos_CU[2], 2)
print('===========================')
print('RELEASE POINT AVG')
print(f"4-Seam Fast Ball : ({avg_ff_x}, {avg_ff_z})")
print(f"Curve : ({avg_cu_x}, {avg_cu_z})")
print(f"X-axis dif : {(abs(avg_ff_x - avg_cu_x) / 6) * 100}%")
print(f"Z-axis dif : {(abs(avg_ff_z - avg_cu_z)/7) * 100}%")

plt.suptitle(f"Yoshinobu Yamamoto: FF vs FS Tunneling Effect", fontsize=16)
plt.show()