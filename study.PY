import pandas as pd 
import oracledb as DB
import numpy as np
import seaborn as sns
import traceback
import matplotlib.pyplot as plt
import folium
import missingno as msno

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

non_ab_events = [
    "walk", "intent_walk", "hit_by_pitch",
    "sac_fly", "sac_bunt", "sac_fly_double_play",
    "catcher_interf" 
]
hit_for_run_events = ["single", "double", "triple", "home_run"]

def hit_distance_analysis(df:pd.DataFrame):
    df = df[
        (df['LAUNCH_SPEED'] >= 40) &
        (df['LAUNCH_ANGLE'] >= -30) & 
        (df['LAUNCH_ANGLE'] <= 55) &
        (df['HIT_DISTANCE_SC'] >= 10)].copy()
    feats = ['HIT_DISTANCE_SC', 'LAUNCH_SPEED', 'LAUNCH_ANGLE']
    df = df.dropna(subset=feats)

    # ===============================================

    X = df[['LAUNCH_SPEED', 'LAUNCH_ANGLE']]
    y = df['HIT_DISTANCE_SC']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    degrees = range(1, 7)
    results = []

    for d in degrees:
        pnr = PolynomialFeatures(degree=d, include_bias=False)
        X_poly_train = pnr.fit_transform(X_train)
        X_poly_test = pnr.transform(X_test)

        lr = LinearRegression().fit(X_poly_train, y_train)
        y_hat = lr.predict(X_poly_test)

        results.append({
            'degree': d,
            'n_features': X_poly_train.shape[1],
            'train_r2': lr.score(X_poly_train, y_train),
            'test_r2': lr.score(X_poly_test, y_test),
            'mse': mean_squared_error(y_test, y_hat),
            'rmse': root_mean_squared_error(y_test, y_hat),
            'mae': mean_absolute_error(y_test, y_hat),
        })

        print(f"  degree={d} | features: {X_poly_train.shape[1]:>3} | "
            f"R²: {results[-1]['test_r2']:.4f} | "
            f"RMSE: {results[-1]['rmse']:.2f} | "
            f"MAE: {results[-1]['mae']:.2f}")

    res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1) R² (Train vs Test)
    axes[0].plot(res['degree'], res['train_r2'], marker='o', color='#3266AD', linewidth=2, label='Train R²')
    axes[0].plot(res['degree'], res['test_r2'], marker='o', color='#D83E23', linewidth=2, label='Test R²')
    axes[0].fill_between(res['degree'], res['train_r2'], res['test_r2'], alpha=0.15, color='red')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('R²')
    axes[0].set_title('R² by Degree (Gap = Overfitting)')
    axes[0].legend()
    axes[0].set_xticks(res['degree'])
    for _, row in res.iterrows():
        axes[0].annotate(f"{row['test_r2']:.3f}", (row['degree'], row['test_r2']),
                        textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9)

    # 2) RMSE & MAE
    axes[1].plot(res['degree'], res['rmse'], marker='s', color='#D83E23', linewidth=2, label='RMSE')
    axes[1].plot(res['degree'], res['mae'], marker='^', color='#F4A261', linewidth=2, label='MAE')
    axes[1].set_xlabel('Polynomial Degree')
    axes[1].set_ylabel('Error (ft)')
    axes[1].set_title('RMSE & MAE by Degree')
    axes[1].legend()
    axes[1].set_xticks(res['degree'])
    for _, row in res.iterrows():
        axes[1].annotate(f"{row['rmse']:.1f}", (row['degree'], row['rmse']),
                        textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9, color='#D83E23')
        axes[1].annotate(f"{row['mae']:.1f}", (row['degree'], row['mae']),
                        textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9, color='#F4A261')

    # 3) 차수별 개선폭
    res['r2_gain'] = res['test_r2'].diff().fillna(0)
    colors = ['#3266AD' if g > 0.03 else '#F4A261' if g > 0 else '#D83E23' for g in res['r2_gain']]
    bars = axes[2].bar(res['degree'], res['r2_gain'], color=colors, edgecolor='white')
    for bar, val in zip(bars, res['r2_gain']):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'+{val:.3f}', ha='center', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Polynomial Degree')
    axes[2].set_ylabel('ΔR² (vs previous degree)')
    axes[2].set_title('Marginal R² Gain per Degree')
    axes[2].set_xticks(res['degree'])
    axes[2].axhline(0, color='black', lw=0.8)

    fig.suptitle('Polynomial Regression: Degree Selection Analysis',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # --------------------------
    # 적합 데이터 수 : 52,626
    # --------------------------

    # from sklearn.preprocessing import PolynomialFeatures
    # from sklearn.linear_model import LinearRegression 
    # from sklearn.model_selection import train_test_split
    
    # pnr = PolynomialFeatures(degree=3, include_bias= False)
    # X = df[['LAUNCH_SPEED', 'LAUNCH_ANGLE']] 
    # y = df['HIT_DISTANCE_SC']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42) 

    # X_poly_train = pnr.fit_transform(X_train)
    # X_poly_test = pnr.transform(X_test)

    # lr = LinearRegression().fit(X_poly_train, y_train)
    # print(lr.score(X_poly_test, y_test))
    # print(lr.coef_)

    # from mpl_toolkits.mplot3d import Axes3D

    # # ── 예측 표면 생성 ──
    # speed_range = np.linspace(X_train['LAUNCH_SPEED'].min(), X_train['LAUNCH_SPEED'].max(), 80)
    # angle_range = np.linspace(X_train['LAUNCH_ANGLE'].min(), X_train['LAUNCH_ANGLE'].max(), 80)
    
    # S, A = np.meshgrid(speed_range, angle_range)

    # grid = pd.DataFrame({'LAUNCH_SPEED': S.ravel(), 'LAUNCH_ANGLE': A.ravel()})
    # Z = np.clip(lr.predict(pnr.transform(grid)), 0, 500).reshape(S.shape)

    # fig = plt.figure(figsize=(14, 9))
    # ax = fig.add_subplot(111, projection='3d')

    # # 예측 표면
    # ax.plot_surface(S, A, Z, cmap='RdYlBu_r', alpha=0.6, edgecolor='none')

    # # 실제 데이터 (테스트셋)
    # ax.scatter(X_test['LAUNCH_SPEED'], X_test['LAUNCH_ANGLE'], y_test,
    #         s=5, alpha=0.4, color='black', label='Actual')

    # ax.set_xlabel('Exit Velocity (mph)')
    # ax.set_ylabel('Launch Angle (°)')
    # ax.set_zlabel('Distance (ft)')
    # ax.set_title('Polynomial Regression Surface (degree=3)', fontsize=14, fontweight='bold')
    # ax.legend()

    # # 초기 시점 각도 조절
    # ax.view_init(elev=25, azim=135)

    # plt.tight_layout()
    # plt.show() 

def Masson_Miller(df:pd.DataFrame): 
    mm_origin = df[df['PITCHER'] == 695243]
    mm_origin['SEASON'] = pd.to_datetime(mm_origin['GAME_DATE']).dt.year
    mm_origin = mm_origin.sort_values(by=['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'], ascending=[True, True, True]).reset_index(drop=True)
    
    # =======================================
    # 1. 구종 통게
    # =======================================
    mm = mm_origin.copy()

    event = mm[mm['PITCH_EVENTS'].notna()]
    event['IS_AB'] = np.where(
        (event['PITCH_EVENTS'].notna()) & ~(event['PITCH_EVENTS'].isin([non_ab_events])),
        1, 0
    )
    event['IS_HIT_FOR_RUN'] = np.where(
        event['PITCH_EVENTS'].isin(hit_for_run_events), 1, 0
    )
    condlist = [
        event['PITCH_EVENTS'] == 'single',
        event['PITCH_EVENTS'] == 'double',
        event['PITCH_EVENTS'] == 'triple',
        event['PITCH_EVENTS'] == 'home_run'
    ]
    event['HIT_FOR_BASE'] = np.select(condlist, [1,2,3,4], default=0)

    event = event.groupby(['PITCH_TYPE', 'SEASON']).agg(
        AB = ('IS_AB', 'sum'),
        HIT_FOR_RUN=('IS_HIT_FOR_RUN', 'sum'),
        RUN_BASE = ('HIT_FOR_BASE', 'sum')
    )
    event['OBP'] = round(event['HIT_FOR_RUN'] / event['AB'], 3)
    event['OSLG'] = round(event['RUN_BASE'] / event['AB'], 3)

    mm = mm.groupby(['PITCH_TYPE', 'SEASON']).agg(
            COUNT = ('RELEASE_SPEED', 'count'),
            SPEED_AVG = ('RELEASE_SPEED', 'mean'),
    )
    mm['SPEED_AVG'] = round(mm['SPEED_AVG'], 1)
    mm['RATIO'] = round(mm['COUNT'] / mm.groupby('SEASON')['COUNT'].transform('sum') * 100, 1).astype(str) + "%"
    res = pd.merge(
        left=mm, right=event, how='inner',
        left_index=True, right_index=True
    )

    print(res)
    mask = (mm_origin['PITCH_TYPE'] == 'FF') & (mm_origin['SEASON'] == 2025)
    subset = mm_origin[mask]

    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax, year, title in zip(axes, [2025, 2026], ['2025 Slider', '2026 Slider']):
        subset = mm_origin[
            (mm_origin['PITCH_TYPE'] == 'SL') & (mm_origin['SEASON'] == year)
        ].dropna(subset=['PLATE_X', 'PLATE_Z'])

        sns.kdeplot(
            data=subset,
            x='PLATE_X', y='PLATE_Z',
            cmap='hot_r',          # 빨강→노랑→흰색 (이미지와 유사)
            fill=True,
            levels=100,             # 등고선 수를 높여서 부드럽게
            thresh=0.2,           # 밀도 낮은 외곽을 자연스럽게 페이드
            ax=ax,
            alpha=0.9,
        )

        # 스트라이크존 사각형 (피트 단위)
        sz = Rectangle(
            (-0.833, 1.5), 1.666, 2.0,
            fill=False, edgecolor='black', linewidth=2, zorder=10
        )
        ax.add_patch(sz)

        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_xlabel('PLATE_X (CAPCHER VIEW)')
        ax.set_ylabel('PLATE_Z')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    mm = mm_origin.copy()
    mm['PREV_PITCH_TYPE'] = mm.groupby(['GAME_PK','AT_BAT_NUMBER'])['PITCH_TYPE'].shift(1)
    transition = pd.crosstab(
        mm['PREV_PITCH_TYPE'],
        mm['PITCH_TYPE'],
        normalize='index'
    )
    print(transition)

def calc_decision_point(df:pd.DataFrame):
    """
    스윙 결정 시점(t=0.25초)에서의 보정된 x, y, z 좌표 계산
    
    Parameters:
        df: PITCH_DESCRIPTION, VX0, VY0, VZ0, AX, AY, AZ, 
            RELEASE_POS_X, RELEASE_POS_Z, PLATE_X, PLATE_Z 컬럼 필요
        t_decision: 스윙 결정 시점 (기본 0.25초)
    
    Returns:
        DataFrame with columns: X_AT_DECISION, Y_AT_DECISION, Z_AT_DECISION
    """
    t_decision=0.25

    # ── 홈플레이트 도달 시간 계산 ──
    # y(t) = RELEASE_POS_Y + VY0*t + 0.5*AY*t² = 0
    a = 0.5 * df['AY']
    b = df['VY0']
    c = df['RELEASE_POS_Y'] - 1.41

    discriminant = b**2 - 4*a*c
    t_plate = (-b - np.sqrt(discriminant)) / (2 * a)

    # ── 궤적 공식으로 예측한 홈플레이트 좌표 ──
    pred_plate_x = df['RELEASE_POS_X'] + df['VX0'] * t_plate + 0.5 * df['AX'] * t_plate**2
    pred_plate_z = df['RELEASE_POS_Z'] + df['VZ0'] * t_plate + 0.5 * df['AZ'] * t_plate**2

    # ── 보정 계수 ──
    error_x = df['PLATE_X'] - pred_plate_x
    error_z = df['PLATE_Z'] - pred_plate_z
    correction_ratio = t_decision / t_plate

    # ── 결정 시점 원시 좌표 ──
    raw_x = df['RELEASE_POS_X'] + df['VX0'] * t_decision + 0.5 * df['AX'] * t_decision**2
    raw_z = df['RELEASE_POS_Z'] + df['VZ0'] * t_decision + 0.5 * df['AZ'] * t_decision**2
    raw_y = df['RELEASE_POS_Y'] + df['VY0'] * t_decision + 0.5 * df['AY'] * t_decision**2

    # ── 보정 적용 ──
    result = pd.DataFrame({
        'X_AT_DECISION': raw_x + error_x * correction_ratio,
        'Y_AT_DECISION': raw_y,
        'Z_AT_DECISION': raw_z + error_z * correction_ratio,
    }, index=df.index)

    return result

def build_sliding_data(df):
        MAX_PITCH = 15
        all_rows = []

        for (gk, ab), group in df.groupby(['GAME_PK', 'AT_BAT_NUMBER']):
            group = group.sort_values('PITCH_NUMBER')

            for t in range(2, len(group) + 1):
                history = group.iloc[:t - 1]
                target = group.iloc[t - 1]

                row = {}

                # ── 이전 투구들의 원시 값 + 패딩 ──
                for i in range(MAX_PITCH - 1):
                    if i < len(history):
                        pitch = history.iloc[i]
                        for col in pitch_features:
                            row[f'p{i+1}_{col}'] = pitch[col]
                        row[f'p{i+1}_EXISTS'] = 1
                    else:
                        for col in pitch_features:
                            row[f'p{i+1}_{col}'] = 0
                        row[f'p{i+1}_EXISTS'] = 0

                # ── 타자 반응 ──
                for i in range(MAX_PITCH - 1):
                    if i < len(history):
                        row[f'p{i+1}_PITCH_DESCRIPTION'] = history.iloc[i]['PITCH_DESCRIPTION']
                    else:
                        row[f'p{i+1}_PITCH_DESCRIPTION'] = 'PAD'

                # ── 이전 투구의 구종 라벨 ──
                for i in range(MAX_PITCH - 1):
                    if i < len(history):
                        row[f'p{i+1}_IS_FF'] = 1 if history.iloc[i]['PITCH_TYPE'] == 'FF' else 0
                        row[f'p{i+1}_IS_SL'] = 1 if history.iloc[i]['PITCH_TYPE'] == 'SL' else 0
                    else:
                        row[f'p{i+1}_IS_FF'] = 0
                        row[f'p{i+1}_IS_SL'] = 0

                # ── 상황 변수 ──
                row['ACTUAL_LENGTH'] = len(history)
                row['BALLS'] = target['BALLS']
                row['STRIKES'] = target['STRIKES']
                row['STAND'] = 1 if target['STAND'] == 'R' else 0

                # ── 타석 식별 키 (GroupKFold용) ──
                row['GAME_AB_KEY'] = f"{gk}_{ab}"

                # ── 예측 대상 ──

                row['target_TYPE'] = target['PITCH_TYPE']
                row['target_PLATE_X'] = target['PLATE_X']
                row['target_PLATE_Z'] = target['PLATE_Z']
                row['target_SPEED'] = target['RELEASE_SPEED']
                row['target_PFX_X'] = target['PFX_X']
                row['target_PFX_Z'] = target['PFX_Z']
                row['target_AX'] = target['AX']
                row['target_AZ'] = target['AZ']
                row['target_RESULT'] = target['PITCH_DESCRIPTION']
                row['target_SZ_TOP'] = 1.5
                row['target_SZ_BOT'] = 3.5

                all_rows.append(row)

        return pd.DataFrame(all_rows)

def my_model(df: pd.DataFrame):
    # 데이터 전처리 
    pitch_features = [
        'RELEASE_SPEED', 'AX', 'AZ',
        'PFX_X', 'PFX_Z',
        'RELEASE_SPIN_RATE', 
        'X_AT_DECISION', 'Y_AT_DECISION', 'Z_AT_DECISION'
    ]

    rhp = df[(df['P_THROWS'] == 'R') & (df['PITCH_TYPE'].isin(['FF', 'SL']))]
    rhp = rhp.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['DID_SWING'] = np.where(
        rhp['PITCH_DESCRIPTION'].isin(
            ['swinging_strike', 'swinging_strike_blocked', 'foul',
             'foul_tip', 'foul_bunt', 'hit_into_play']
        ), 1, 0
    )

    col = calc_decision_point(df)
    rhp = pd.concat([rhp, col], axis=1)
    rhp = rhp.dropna(subset=pitch_features)

    # ── PREV 원시값 ──
    group = rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])
    prev_cols = ['RELEASE_SPEED', 'AX', 'AZ', 'X_AT_DECISION', 'Z_AT_DECISION', 'PLATE_X', 'PLATE_Z']
    prev_df = group[prev_cols].shift(1).add_prefix('PREV_')
    rhp = pd.concat([rhp, prev_df], axis=1)

    # ── 궤적 괴리 비율 ──
    rhp['MOVE_RATE'] = (rhp['PLATE_X'] - rhp['X_AT_DECISION']) / rhp['X_AT_DECISION']
    rhp['DROP_RATE'] = (rhp['Z_AT_DECISION'] - rhp['PLATE_Z']) / rhp['Z_AT_DECISION']

    # ── 직전 대비 변화량 ──
    rhp['X_AT_DECISION_DIFF'] = rhp['X_AT_DECISION'] - rhp['PREV_X_AT_DECISION']
    rhp['Z_AT_DECISION_DIFF'] = rhp['Z_AT_DECISION'] - rhp['PREV_Z_AT_DECISION']

    # ── NaN 처리: PREV는 전체 평균으로, DIFF는 0으로 ──
    prev_fill_cols = ['PREV_RELEASE_SPEED', 'PREV_AX', 'PREV_AZ',
                      'PREV_X_AT_DECISION', 'PREV_Z_AT_DECISION',
                      'PREV_PLATE_X', 'PREV_PLATE_Z']
    for c in prev_fill_cols:
        original = c.replace('PREV_', '')
        rhp[c] = rhp[c].fillna(rhp[original].mean())

    rhp['X_AT_DECISION_DIFF'] = rhp['X_AT_DECISION_DIFF'].fillna(0)
    rhp['Z_AT_DECISION_DIFF'] = rhp['Z_AT_DECISION_DIFF'].fillna(0)
    rhp['MOVE_RATE'] = rhp['MOVE_RATE'].fillna(0)
    rhp['DROP_RATE'] = rhp['DROP_RATE'].fillna(0)

    # ── 존 안/밖 판별 (타자 성향 계산용) ──
    DZ_TOP = 4.69
    DZ_BOT = 3.76
    DZ_EDGE = 0.63

    rhp['IN_ZONE_AT_DECISION'] = (
        (abs(rhp['X_AT_DECISION']) <= DZ_EDGE) &
        (rhp['Z_AT_DECISION'] >= DZ_BOT) &
        (rhp['Z_AT_DECISION'] <= DZ_TOP)
    ).astype(int)
    
    # 정렬 (타자별 시간순)
    rhp = rhp.sort_values(['BATTER', 'GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])

    # ── 1. 타자 누적 스윙율 (과거 이력만) ──
    rhp['BATTER_CUM_SWING'] = (
        rhp.groupby('BATTER')['DID_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # ── 2. 존 안/밖 별 스윙율 ──
    # 존 안 공에 대한 스윙율 (공격성)
    rhp['IN_ZONE_SWING'] = rhp['IN_ZONE_AT_DECISION'] * rhp['DID_SWING']
    rhp['BATTER_ZONE_SWING'] = (
        rhp.groupby('BATTER')['IN_ZONE_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # 존 밖 공에 대한 스윙율 (선구안)
    rhp['OUT_ZONE_SWING'] = (1 - rhp['IN_ZONE_AT_DECISION']) * rhp['DID_SWING']
    rhp['BATTER_CHASE_RATE'] = (
        rhp.groupby('BATTER')['OUT_ZONE_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # ── 3. 해당 타석 내 스윙율 (현재 타석의 적극성) ──
    rhp = rhp.sort_values(['GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['AB_CUM_SWING'] = (
        rhp.groupby(['GAME_PK', 'AT_BAT_NUMBER'])['DID_SWING']
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=[0, 1], drop=True)
    )

    # ── 4. 타자의 누적 투구 수 (경험치) ──
    rhp = rhp.sort_values(['BATTER', 'GAME_PK', 'AT_BAT_NUMBER', 'PITCH_NUMBER'])
    rhp['BATTER_PITCH_COUNT'] = rhp.groupby('BATTER').cumcount()

    # ── NaN 처리 ──
    global_swing = rhp['DID_SWING'].mean()
    rhp['BATTER_CUM_SWING'] = rhp['BATTER_CUM_SWING'].fillna(global_swing)
    rhp['BATTER_ZONE_SWING'] = rhp['BATTER_ZONE_SWING'].fillna(global_swing)
    rhp['BATTER_CHASE_RATE'] = rhp['BATTER_CHASE_RATE'].fillna(global_swing)
    rhp['AB_CUM_SWING'] = rhp['AB_CUM_SWING'].fillna(global_swing)

    # 중간 계산용 컬럼 제거
    rhp = rhp.drop(columns=['IN_ZONE_SWING', 'OUT_ZONE_SWING', 'PLATE_X', 'PLATE_Z'])

    # ── 최종 feature 리스트 ──
    model_features = [
        'RELEASE_SPEED', 'AX', 'AZ',
        'X_AT_DECISION', 'Z_AT_DECISION',
        'MOVE_RATE', 'DROP_RATE',
        'X_AT_DECISION_DIFF', 'Z_AT_DECISION_DIFF',
        'PREV_RELEASE_SPEED', 'PREV_AX', 'PREV_AZ',
        'PREV_X_AT_DECISION', 'PREV_Z_AT_DECISION',
        'STRIKES', 'BALLS',
        'PREV_PLATE_X', 'PREV_PLATE_Z',
        # 타자 관련 features 
        'BATTER_ZONE_SWING', 'BATTER_CHASE_RATE'
    ]

    non_miller_df = rhp[rhp['PITCHER'] != 695243].copy()
    miller_df = rhp[rhp['PITCHER'] == 695243].copy()

    # 밀러 데이터 분리 
    X_miller = miller_df[model_features].copy()
    X_miller['STAND'] = (miller_df['STAND'] == 'R').astype(int)
    y_miller = miller_df['DID_SWING']

    X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
        X_miller, y_miller, test_size=.3, random_state=42
    )

    # 베이스 학습 : non-밀러 전체 + 밀러 train 데이터
    base_df = pd.concat([non_miller_df, miller_df.loc[X_m_train.index]])

    X = base_df[model_features].copy()
    X['STAND'] = (base_df['STAND'] == 'R').astype(int)
    y = base_df['DID_SWING']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)

    from lightgbm import LGBMClassifier

    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=.1,
        num_leaves=31,
        min_child_samples=50,
        subsample=.7,
        reg_alpha=1.0,
        reg_lambda=.1,
        colsample_bytree=.9,
        random_state=42,
        verbose=-1
    )

    lgbm.fit(X_train, y_train)
    y_hat = lgbm.predict(X_test)

    print(" === Base 모델 평가 지표 === ")
    print(f"Train score: {lgbm.score(X_train, y_train):.4f}")
    print(f"Test score:  {lgbm.score(X_test, y_test):.4f}")
    print(metrics.classification_report(y_test, y_hat))

    # ===================================== 
    # 전이 학습 (base → Miller fine-tune)   
    # =====================================

    fine_model = LGBMClassifier(
        n_estimators=50,        # 적은 트리로 미세 조정
        max_depth=4,
        learning_rate=0.01,     # 작은 보폭
        num_leaves=31,
        min_child_samples=10,   # Miller 데이터가 적으니 완화
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    fine_model.fit(X_m_train, y_m_train, init_model=lgbm)
    y_hat_fine = fine_model.predict(X_m_test)
    print("=== 전이 학습 (fine-tune) ===")
    print(f"Score: {fine_model.score(X_m_test, y_m_test):.4f}")
    print(metrics.classification_report(y_m_test, y_hat_fine))

if __name__ == "__main__":
    try:
        conn = DB.connect(user="C##AKSEN", password="1234", dsn="localhost:1521/xe")
        cursor = conn.cursor()

        sql = """
            SELECT *
            FROM TMP_TABLE
        """

        cursor.execute(sql)
        df = pd.DataFrame(
            data= cursor.fetchall(), 
            columns= [d[0] for d in cursor.description]
        )

        # hit_distance_analysis(df)
        # pfx_z_analysis(df)
        # pitch_type_classification(df)
        # Masson_Miller(df)    
        my_model(df)


    except Exception as e:
        traceback.print_exc()
    finally:
        if "conn" in locals():
            conn.close()