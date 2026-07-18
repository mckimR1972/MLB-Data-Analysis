import oracledb as DB 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import xgboost as xgb
import shap 

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report


def pitch_tunnel_cal(row: pd.Series, tmp_dict: dict):
    """
    투구의 물리적 궤적을 계산하여 25피트 지점(터널 포인트)의 X, Z 좌표를 반환합니다.
    """
    target_x, target_z = row['PLATE_X'], row['PLATE_Z']
    
    # 피칭 터널 시작 - 포구지점 
    sqrt = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 1.417)) # home_plate_y = 1.417 
    end_t = (-row['VY0'] - sqrt) / row['AY'] 
    t = np.linspace(0, end_t, 100)

    theorem_x = row['RELEASE_POS_X'] + row['VX0']*t + .5*row['AX']*t**2
    y = row['RELEASE_POS_Y'] + row['VY0']*t + .5*row['AY']*t**2
    theorem_z = row['RELEASE_POS_Z'] + row['VZ0']*t + .5*row['AZ']*t**2
    
    error_x = target_x - theorem_x[-1]
    error_z = target_z - theorem_z[-1]

    # 비행 시간의 비율(0~1)에 따라 오차를 점진적으로 배분
    correction_factor = (t / end_t)

    # 25피트 지점에서 z좌표 위치 계산
    sqrt25 = np.sqrt(row['VY0'] ** 2 - 2 * row['AY'] * (row['RELEASE_POS_Y'] - 25))
    t25 = (-row['VY0'] - sqrt25) / row['AY']
    
    # Y좌표가 25feet일 때 t값으로 그 때 Z좌표 구하기 
    z25 = row['RELEASE_POS_Z'] + row['VZ0'] * t25 + 0.5*row['AZ']*t25**2
    z25 = z25 + (error_z * t25 / end_t)
    
    x25 = row['RELEASE_POS_X'] + row['VX0'] * t25 + 0.5*row['AX']*t25**2
    x25 = x25 + (error_x * t25 / end_t)

    tmp_dict['Z_FOR_Y25'].append(z25)
    tmp_dict['X_FOR_Y25'].append(x25)


def test_function(df: pd.DataFrame):
    """
    데이터 전처리, 시계열 피처 엔지니어링, 모델 학습 및 평가를 수행합니다.
    """
    # 1. 정렬 (시계열 분석의 기본)
    df = df.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER']).reset_index(drop=True)
    
    # 2. 물리 궤적(터널 포인트) 계산
    tmp_dict = {'X_FOR_Y25': [], 'Z_FOR_Y25':[]}
    for _, row in df.iterrows():
        pitch_tunnel_cal(row, tmp_dict)
        
    tmp_df = pd.DataFrame(tmp_dict)
    df = pd.concat([df, tmp_df], axis=1)

    # 3. 시계열(Lag) 변수 생성: 1구 전 데이터
    group_col = ['GAME_PK', 'AT_BAT_NUMBER']
    df['PREV_PITCH_TYPE'] = df.groupby(group_col)['PITCH_TYPE'].shift(1).fillna('START')
    df['PREV_X_FOR_Y25'] = df.groupby(group_col)['X_FOR_Y25'].shift(1).fillna(0)
    df['PREV_Z_FOR_Y25'] = df.groupby(group_col)['Z_FOR_Y25'].shift(1).fillna(0)
    df['PREV_PLATE_X'] = df.groupby(group_col)['PLATE_X'].shift(1).fillna(0)
    df['PREV_PLATE_Z'] = df.groupby(group_col)['PLATE_Z'].shift(1).fillna(0)
    
    # 4. 시계열(Lag) 변수 생성: 2구 전 데이터 (깊은 기억)
    df['PREV2_PITCH_TYPE'] = df.groupby(group_col)['PITCH_TYPE'].shift(2).fillna('START')
    df['PREV2_PLATE_X'] = df.groupby(group_col)['PLATE_X'].shift(2).fillna(0)
    df['PREV2_PLATE_Z'] = df.groupby(group_col)['PLATE_Z'].shift(2).fillna(0)

    # 5. 시계열 변화량(Differential) 및 누적 피처
    df['SEQ_NUM'] = df['PITCH_NUMBER']
    df['PREV_SPEED'] = df.groupby(group_col)['RELEASE_SPEED'].shift(1).fillna(df['RELEASE_SPEED'])
    df['SPEED_DIFF'] = df['RELEASE_SPEED'] - df['PREV_SPEED']
    df['LOC_DIST_FROM_PREV'] = np.sqrt(
        (df['PLATE_X'] - df['PREV_PLATE_X'])**2 + 
        (df['PLATE_Z'] - df['PREV_PLATE_Z'])**2
    ).fillna(0)

    # 1구 전 데이터가 없는 초구(1구)는 패턴 분석을 위해 제거
    df = df[df['PREV_PITCH_TYPE'] != 'START'].copy()

    # 6. 타겟(Target) 변수 생성: 헛스윙, 루킹 스트라이크, 각종 아웃
    df['RESULT'] = np.where(
        df['PITCH_DESCRIPTION'].isin(['swinging_strike', 'swinging_strike_blocked', 'called_strike']) | 
        df['PITCH_EVENTS'].isin([
            'field_out', 'strikeout', 'grounded_into_double_play', 
            'force_out', 'fielders_choice_out', 'double_play', 
            'fielders_choice', 'strikeout_double_play', 'triple_play'
        ]), 1, 0
    )

    # 7. 원핫 인코딩(더미 변수) 처리
    pitch_type_dummies = pd.get_dummies(df['PITCH_TYPE'], prefix='PT', drop_first=True)
    prev1_pitch_dummies = pd.get_dummies(df['PREV_PITCH_TYPE'], prefix='PPT', drop_first=True)
    prev2_pitch_dummies = pd.get_dummies(df['PREV2_PITCH_TYPE'], prefix='P2PT', drop_first=True)
    
    # 'START' 더미 칼럼 제거 (필요 없는 노이즈 방지)
    prev2_pitch_dummies = prev2_pitch_dummies[[c for c in prev2_pitch_dummies.columns if 'START' not in c]]
    
    strikes_dummies = pd.get_dummies(df['STRIKES'].astype(int), prefix='ST', drop_first=True)
    balls_dummies = pd.get_dummies(df['BALLS'].astype(int), prefix='BL', drop_first=True)
    stand_dummies = pd.get_dummies(df['STAND'], prefix='BAT', drop_first=True)

    # 데이터 결합
    df = pd.concat([
        df, pitch_type_dummies, prev1_pitch_dummies, prev2_pitch_dummies, 
        strikes_dummies, balls_dummies, stand_dummies
    ], axis=1)

    # 8. 모델 투입용 Features 리스트 최종 정리
    features = (
        ['PLATE_X', 'PLATE_Z', 'X_FOR_Y25', 'Z_FOR_Y25', 'RELEASE_SPEED'] +
        ['PREV_PLATE_X', 'PREV_PLATE_Z', 'PREV_X_FOR_Y25', 'PREV_Z_FOR_Y25'] +
        ['PREV2_PLATE_X', 'PREV2_PLATE_Z'] +
        ['SPEED_DIFF', 'LOC_DIST_FROM_PREV', 'SEQ_NUM'] +
        list(pitch_type_dummies.columns) +
        list(prev1_pitch_dummies.columns) +
        list(prev2_pitch_dummies.columns) +
        list(strikes_dummies.columns) +
        list(balls_dummies.columns) +
        list(stand_dummies.columns)
    )
    target = 'RESULT'

    # 9. 모델링 세팅
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=1.3
    )
    
    print("\n모델 학습을 시작합니다...")
    xgb_model.fit(X_train, y_train)

    # 10. 예측 및 성능 평가
    y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
    threshold = 0.4
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\nAccuracy (정확도): {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # 11. SHAP 플롯 시각화
    print("\nSHAP 분석을 준비 중입니다...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    # Feature Importance (기본 XGBoost Gain)
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=15, title='Top 15 Features (Gain)')
    plt.tight_layout()
    plt.show()

    # SHAP Dependence Plot
    # 상호작용 분석 대상 변수가 데이터에 있는지 확인 (없으면 에러 방지)
    interaction_target = "PT_SL" if "PT_SL" in features else None
    
    shap.dependence_plot(
        "PLATE_X", 
        shap_values, 
        X_test.values, 
        feature_names=features,
        interaction_index=interaction_target,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: PLATE_X vs {interaction_target}')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # DB 연결 정보
    conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
    cursor = conn.cursor() 
    
    PITCHER_ID = 695243 # 메이슨 밀러
    
    sql = f""" 
        SELECT * FROM TMP_TABLE
        WHERE PITCHER = {PITCHER_ID}
        ORDER BY GAME_PK, AT_BAT_NUMBER, PITCH_NUMBER
    """
    print('Accessing Oracle DB.....\n')
    cursor.execute(sql)

    cols = [row[0] for row in cursor.description]
    raw_df = pd.DataFrame(data=cursor.fetchall(), columns=cols)
    
    # 궤적 계산에 필요한 필수 컬럼들 결측치 제거
    all_pitch_df = raw_df.dropna(
        subset=['RELEASE_SPEED','PLATE_X', 'PLATE_Z', 
                'RELEASE_POS_X', 'RELEASE_POS_Y', 'RELEASE_POS_Z',
                'VX0', 'VY0', 'VZ0', 'AX', 'AY', 'AZ', 
                'PITCH_TYPE', 'PFX_X', 'PFX_Z', 'STAND']
    ).copy()
    
    # DB에서 가져올 때 발생할 수 있는 대소문자 및 공백 처리
    all_pitch_df['PITCH_DESCRIPTION'] = all_pitch_df['PITCH_DESCRIPTION'].astype(str).str.lower().str.strip()
    all_pitch_df['PITCH_EVENTS'] = all_pitch_df['PITCH_EVENTS'].astype(str).str.lower().str.strip()

    print('All Data Needed Fetched And Setting Had Done....\n')

    # 통합 메인 분석 함수 실행
    test_function(all_pitch_df)

    print('Process Successfully Done!!')

    cursor.close()
    conn.close()
    print('Disconnected With Oracle DB And Close The Program\n')