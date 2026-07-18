import pandas as pd
import numpy as np
import oracledb as db 
import seaborn as sns 
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
# plt.rcParams['font.family'] = 'AppleGothic'  # Mac
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지

conn = db.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
cursor = conn.cursor()

sql = """
    SELECT * 
    FROM KBO_2026
"""

cursor.execute(sql)
raw_df = pd.DataFrame(data=cursor.fetchall(), columns= [name[0] for name in cursor.description])
raw_df = raw_df.sort_values(['AT_BAT_NUMBER', 'PITCH_NUMBER'])
raw_df = raw_df[raw_df['BATTER'] == '1번타자 이주형']
sz_bot, sz_top = raw_df['SZ_BOT'].mean(), raw_df['SZ_TOP'].mean()

fig=plt.figure(figsize=(9, 6))
plot = sns.scatterplot(
    data= raw_df, x= 'PLATE_X', y='PLATE_Z', hue='PITCH_TYPE', s= 45 
)
plot.set_xlim(-2.5, 2.5)
plot.set_ylim(0, 5)
border_line = plt.Rectangle(
    [-0.83, sz_bot], 1.66, sz_top - sz_bot, linewidth=1.5, color= 'black', fill=False
)
plot.add_patch(border_line)
plot.set_box_aspect(1)
plt.show()