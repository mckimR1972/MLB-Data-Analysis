import requests
import json
import pandas as pd
import numpy as np
import time

game_id = "20260401KTHH02026"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": f"https://m.sports.naver.com/game/{game_id}/relay"
}

# 투수 pcode -> 이름 매핑 테이블 만들기
pitcher_names = {}
first_res = requests.get(
    f"https://api-gw.sports.naver.com/schedule/games/{game_id}/relay?inning=1",
    headers=headers
)
if first_res.status_code == 200:
    relay_data = first_res.json()["result"]["textRelayData"]
    for src in ["homeLineup", "awayLineup", "homeEntry", "awayEntry"]:
        for p in relay_data.get(src, {}).get("pitcher", []):
            pitcher_names[p["pcode"]] = p["name"]

time.sleep(0.5)

# 전체 이닝 데이터 수집 (중복 제거)
all_relays = []
seen_nos = set()

for i in range(1, 10):
    url = f"https://api-gw.sports.naver.com/schedule/games/{game_id}/relay?inning={i}"
    res = requests.get(url, headers=headers)
    
    if res.status_code == 200:
        data = res.json()
        relays = data.get("result", {}).get("textRelayData", {}).get("textRelays", [])
        
        count = 0
        for relay in relays:
            key = (relay["inn"], relay["no"])
            if key not in seen_nos:
                seen_nos.add(key)
                all_relays.append(relay)
                count += 1
        
        print(f"{i}회 수집 완료 - 신규 {count}개 / 전체 {len(relays)}개")
    else:
        print(f"{i}회 실패 - status: {res.status_code}")
    
    time.sleep(0.5)

# crossPlateZ 계산 함수
def calc_plate_z(row):
    try:
        a = 0.5 * row["ay"]
        b = row["vy0"]
        c = row["y0"] - 0.7083
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        t = (-b - np.sqrt(discriminant)) / (2*a)
        plate_z = row["z0"] + row["vz0"] * t + 0.5 * row["az"] * t**2
        return round(plate_z, 3)
    except:
        return None

# 투구 데이터 추출
pitches = []

for relay in all_relays:
    if relay["titleStyle"] == "0":
        continue
    
    batter = relay["title"]
    inn = relay["inn"]
    home_or_away = relay["homeOrAway"]
    
    # 타석 최종 결과 찾기 (type == 13)
    at_bat_result = ""
    for opt in relay["textOptions"]:
        if opt.get("type") == 13:
            at_bat_result = opt["text"]
            break
    
    pts_map = {}
    for pts in relay.get("ptsOptions", []):
        pts_map[pts["pitchId"]] = pts
    
    for opt in relay["textOptions"]:
        if "pitchNum" not in opt:
            continue
        
        pitch_id = opt.get("ptsPitchId", "")
        pts = pts_map.get(pitch_id, {})
        
        # 투수 정보
        pitcher_code = opt.get("currentGameState", {}).get("pitcher", "")
        pitcher_name = pitcher_names.get(pitcher_code, pitcher_code)
        
        # 당시 스코어
        game_state = opt.get("currentGameState", {})
        home_score = game_state.get("homeScore", "0")
        away_score = game_state.get("awayScore", "0")
        
        pitches.append({
            "이닝": inn,
            "공수": "한화공격" if home_or_away == "1" else "키움공격",
            "투수": pitcher_name,
            "타자": batter,
            "타석결과": at_bat_result,
            "홈스코어": home_score,
            "원정스코어": away_score,
            "투구번호": opt["pitchNum"],
            "구속": int(opt["speed"]) if opt.get("speed") else None,
            "구종": opt.get("stuff"),
            "결과코드": opt.get("pitchResult"),
            "텍스트": opt.get("text"),
            "pitchId": pitch_id,
            "crossPlateX": pts.get("crossPlateX"),
            "topSz": pts.get("topSz"),
            "bottomSz": pts.get("bottomSz"),
            "vx0": pts.get("vx0"),
            "vy0": pts.get("vy0"),
            "vz0": pts.get("vz0"),
            "ax": pts.get("ax"),
            "ay": pts.get("ay"),
            "az": pts.get("az"),
            "x0": pts.get("x0"),
            "y0": pts.get("y0"),
            "z0": pts.get("z0"),
            "stance": pts.get("stance"),
        })

df = pd.DataFrame(pitches)
df['스코어'] = df['원정스코어'].astype(str) + '-' + df['홈스코어'].astype(str)
result_map = {"B": "볼", "S": "헛스윙", "T": "스트라이크", "F": "파울", "H": "타격"}
df["결과"] = df["결과코드"].map(result_map)

# crossPlateZ 계산
df["crossPlateZ"] = df.apply(calc_plate_z, axis=1)


import oracledb as DB 

conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
cursor = conn.cursor()

# target_df = df[df['투수'] == '김광현']
target_df = df.copy()
for _, row in target_df.iterrows():
    tmp = []
    
    tmp.append(str(row['투수']))
    tmp.append(str(row['타자']))
    tmp.append(int(row['투구번호']))
    tmp.append(str(row['구종']))
    tmp.append(round(float(row['crossPlateX']), 3))
    tmp.append(round(float(row['crossPlateZ']), 3))  # crossPlateY -> crossPlateZ로 변경
    tmp.append(int(row['구속']))
    tmp.append(round(float(row['x0']), 3))
    tmp.append(round(float(row['y0']), 3))
    tmp.append(round(float(row['z0']), 3))
    tmp.append(round(float(row['vx0']), 3))
    tmp.append(round(float(row['vy0']), 3))
    tmp.append(round(float(row['vz0']), 3))
    tmp.append(round(float(row['ax']), 3))
    tmp.append(round(float(row['ay']), 3))
    tmp.append(round(float(row['az']), 3))
    tmp.append(str(row['결과']))
    tmp.append(str(row['타석결과']))
    tmp.append(str(row['결과코드']))
    tmp.append(round(float(row['bottomSz']), 3))
    tmp.append(round(float(row['topSz']), 3))
    tmp.append(str(row['stance']))
    tmp.append(str(row['이닝']))
    tmp.append(str(row['공수']))
    tmp.append(str(row['스코어']))  # '으악' 대신 타석번호
    tmp.append(int(0))
    sql = """
    insert into KBO_2026 
        values(:1,:2,:3,:4,:5,:6,:7,:8,:9,:10,
        :11,:12,:13,:14,:15,:16,:17,:18,:19,:20,
        :21,:22,:23,:24,:25,:26)
    """

    cursor.execute(sql, tmp)
conn.commit()
cursor.close()
conn.close()