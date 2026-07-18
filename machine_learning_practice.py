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

def pitch_tunnel_cal(row, tmp_dict):
    target_x, target_z = row['PLATE_X'], row['PLATE_Z']
    
    # 비행 시간 및 궤적 계산
    v_y0_sq = row['VY0']**2
    y_rel = row['RELEASE_POS_Y']
    
    # 홈플레이트 도달 시간 (y=1.417)
    sqrt_home = np.sqrt(v_y0_sq - 2 * row['AY'] * (y_rel - 1.417))
    end_t = (-row['VY0'] - sqrt_home) / row['AY']
    
    # 25피트 지점 통과 시간
    sqrt_25 = np.sqrt(v_y0_sq - 2 * row['AY'] * (y_rel - 25))
    t25 = (-row['VY0'] - sqrt_25) / row['AY']
    
    # 오차 보정 (실제 투구 위치와 물리 공식 결과의 차이 배분)
    def get_pos_at_t(t, v0, a, pos0, error, total_t):
        return pos0 + v0*t + 0.5*a*t**2 + (error * t / total_t)

    theory_x_end = row['RELEASE_POS_X'] + row['VX0']*end_t + 0.5*row['AX']*end_t**2
    theory_z_end = row['RELEASE_POS_Z'] + row['VZ0']*end_t + 0.5*row['AZ']*end_t**2
    
    error_x = target_x - theory_x_end
    error_z = target_z - theory_z_end
    
    x25 = get_pos_at_t(t25, row['VX0'], row['AX'], row['RELEASE_POS_X'], error_x, end_t)
    z25 = get_pos_at_t(t25, row['VZ0'], row['AZ'], row['RELEASE_POS_Z'], error_z, end_t)
    
    tmp_dict['Z_FOR_Y25'].append(z25)
    tmp_dict['X_FOR_Y25'].append(x25)
    tmp_dict['Z_PREDICT'].append(theory_z_end)
    tmp_dict['X_PREDICT'].append(theory_x_end)


def analysis(df:pd.DataFrame):
    df = df.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    pitch_type_cnt = df.groupby(['PITCH_TYPE'])['PLATE_X'].count().reset_index(name='COUNT')
    pitch_type_cnt['IS_ABUNDANT'] = np.where(pitch_type_cnt['COUNT'] > 30, 1, 0)

    df = pd.merge(
        left=df, right=pitch_type_cnt, how='inner',
        left_on='PITCH_TYPE', right_on='PITCH_TYPE'
    )
    df['IS_EFFECTIVE'] = np.where(
        df['PITCH_DESCRIPTION'].isin(['called_strike', 'swing_strike', 'swing_strike_blocked', 'foul']) |
        df['PITCH_EVENTS'].isin(['strikeout', 'field_out', 'grounded_into_double_play', 
                                 'force_out', 'double_play', 'strikeout_double_play', 'triple_play']), 
        1, 0
    )
    df = df[df['IS_ABUNDANT'] == 1]

    print(len(df))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot = sns.kdeplot(
        data=df[df['PITCH_TYPE'] == 'FF'], x='SPIN_AXIS', fill=True, hue= 'IS_EFFECTIVE', 
        ax=axes[0]
    )
    plot1 = sns.kdeplot(
        data=df[df['PITCH_TYPE'] == 'SL'], x='SPIN_AXIS', fill=True, hue= 'IS_EFFECTIVE', 
        ax=axes[1]
    )
    
    import xgboost as xgb 
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

    st_dummies = pd.get_dummies(df['STRIKES'], prefix='ST', drop_first=True)
    df = pd.concat([df, st_dummies], axis=1)

    features = ['RELEASE_SPEED', 'SPIN_AXIS',
                'RELEASE_POS_X', 'RELEASE_POS_Z', 'RELEASE_POS_Y',
                'PLATE_X', 'PLATE_Z', 
                'PFX_X', 'PFX_Z', 
                'ST_1', 'ST_2']
    target = 'IS_EFFECTIVE'

    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)

    model = xgb.XGBClassifier(
        n_estimators=1000, 
        learning_rate=0.05, 
        max_depth=6, 
        subsample=0.8, 
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    xgb.plot_importance(model, importance_type='gain')

    y_pred = model.predict(X_test)
    # predict_proba: 스트라이크를 잡거나 아웃시킬 '확률(%)' 예측 (ROC-AUC 계산에 필요)
    y_pred_proba = model.predict_proba(X_test)[:, 1] 

    # 3. 분류 성능 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy (정확도): {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # 4. 상세 분류 리포트 (정밀도, 재현율 확인)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    import shap 
    # 1. SHAP Explainer 계산
    explainer = shap.TreeExplainer(model)
    # SHAP 계산 시에도 .values를 쓰는 것이 안전합니다.
    shap_values = explainer.shap_values(X_test)

    # 2. 교호작용 시각화 (에러 수정 포인트)
    # 데이터는 .values로 전달하고, feature_names를 직접 지정해 인덱스 충돌을 방지합니다.
    shap.dependence_plot(
        "RELEASE_SPEED", 
        shap_values, 
        X_test.values,         # .values 추가 (numpy 배열로 전달)
        feature_names=features, # 원래 컬럼명 리스트 전달
        interaction_index="PLATE_Z"
    )

    # # 선형회귀 테스트 
    # import statsmodels.api as sm
    # # 비선형관계 분석 가능  
    # import xgboost as xgb
    # from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LinearRegression
    
    # features = ['HC_X', 'HC_Y', 'LAUNCH_ANGLE', 'LAUNCH_SPEED']
    # target = 'XW_OBA'
    
    # X, y = df[features], df[target]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # # 학습 
    # X_train_with_const = sm.add_constant(X_train)
    # model = xgb.XGBRegressor(
    #     n_estimators=1000, 
    #     learning_rate=0.05, 
    #     max_depth=6, 
    #     subsample=0.8, 
    #     colsample_bytree=0.8,
    #     n_jobs=-1,
    #     random_state=42
    # )
    # model.fit(X_train, y_train)
    # xgb.plot_importance(model, importance_type='gain')
    # # 테스트 
    # from sklearn.metrics import mean_squared_error, r2_score
    
    # y_pred = model.predict(X_test)

    # # 평가 지표 계산
    # mse = mean_squared_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)

    # print(f"XGBOOST MSE: {mse:.4f}")
    # print(f"XGBOOST R-SQUARE SCORE: {r2:.4f}")

    # plt.figure(figsize=(10, 6))
    # xgb.plot_importance(model, importance_type='weight')
    
    # import shap

    # # 1. SHAP Explainer 계산
    # explainer = shap.TreeExplainer(model)
    # # SHAP 계산 시에도 .values를 쓰는 것이 안전합니다.
    # shap_values = explainer.shap_values(X_test)

    # # 2. 교호작용 시각화 (에러 수정 포인트)
    # # 데이터는 .values로 전달하고, feature_names를 직접 지정해 인덱스 충돌을 방지합니다.
    # shap.dependence_plot(
    #     "LAUNCH_SPEED", 
    #     shap_values, 
    #     X_test.values,         # .values 추가 (numpy 배열로 전달)
    #     feature_names=features, # 원래 컬럼명 리스트 전달
    #     interaction_index="LAUNCH_ANGLE"
    # )

    plt.show()

    # 모델의 부스터 객체를 가져와서 텍스트 파일로 저장
    # booster = model.get_booster()
    # booster.dump_model('xgboost_model_rules.txt')

    # print("모델의 내부 규칙이 'xgboost_model_rules.txt' 파일로 저장되었습니다!")

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="020304", dsn="localhost:1521/xe")
        cursor = conn.cursor()
        
        # 데이터 로드
        sql = """
            SELECT * 
            FROM TMP_TABLE 
            WHERE 
                PITCHER = 621242
            ORDER BY GAME_PK, AT_BAT_NUMBER, PITCH_NUMBER 
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
                    'PFX_X', 'PFX_Z', 'SPIN_AXIS'
                ]
        )
        #print(len(clean_df))
        
        analysis(clean_df)


    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'conn' in locals(): conn.close()